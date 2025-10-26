import torch
import torch.nn as nn
import einops
import glob

class ValueCompressor(nn.Module):
    def __init__(self, quant_bits, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.quant_bits = quant_bits
        self.code_bit = 2
        self.code_range = 2 ** self.code_bit
        self.n_e = int(self.feat_dim * self.quant_bits) // self.code_bit
        self.embed_dim = self.n_e * self.code_range
        self.embedding = torch.nn.Parameter(torch.randn(self.n_e * self.code_bit, self.feat_dim))
        self.norm1 = nn.LayerNorm(self.feat_dim)
        self.intermediate_dim = min(4096, self.embed_dim) if self.embed_dim != 6144 else self.embed_dim
        self.enc1 = nn.Linear(self.feat_dim, self.intermediate_dim)
        self.act1 = nn.GELU()
        self.out = nn.Linear(self.intermediate_dim, self.embed_dim)
        self.eps = 1e-6
        self.learnable_scale = torch.nn.Parameter(torch.ones(self.feat_dim))
        self.learnable_scale.requires_grad = False  # freeze learnable scale
        self.CONST = torch.arange(8, dtype=torch.uint8)
        self.CONST1 = (2 ** self.CONST)
        self.CONST2 = (1 << self.CONST)

    def encode(self, x):
        original_bits = x.shape[-1] * 16
        compressed_bit = self.embedding.shape[0]
        compression_ratio = original_bits // compressed_bit
        assert int(16 / self.quant_bits) == compression_ratio
        x = x / (self.learnable_scale + self.eps)
        prescale = torch.norm(x, dim=-1, keepdim=True)
        x = x / (prescale + self.eps)
        x = self.norm1(x)
        x = self.enc1(x)
        x = self.act1(x)
        x = self.out(x)
        x = self.quantize(x)
        return x, prescale

    def quantize(self, x):
        B, S, E = x.shape
        x = x.reshape(B, S, self.n_e, self.code_range)
        x = torch.nn.functional.gumbel_softmax(x, tau=1.0, hard=True)
        x[..., :2] += x[..., 2:3]
        x = x[..., :2].flatten(2)
        return x

    def decode(self, x, prescale):
        x = torch.matmul(x, self.embedding)
        postscale = torch.norm(x, dim=-1, keepdim=True)
        x = x / (postscale + self.eps)
        x = x * (prescale + self.eps)
        x = x * (self.learnable_scale + self.eps)
        return x

    def pack(self, x, prescale, offload=False):
        # assert offload
        shape = x.shape
        dtype = x.dtype
        x = x.flatten().to(torch.uint8)
        x, padding = self.compress_binary_tensor(x)
        assert padding == 0
        data = {
            "code": x,
            "padding": padding,
            "shape": list(shape),
            "prescale": prescale,
            "dtype": dtype,
        }
        if offload:
            data["code"] = data["code"].to("cpu", non_blocking=True)
        return data

    def unpack(self, data, device, offload=False):
        # assert offload
        assert data["code"].dtype == torch.uint8
        if offload:
            code = data["code"].to(device)
        else:
            code = data["code"]
        x = self.decompress_binary_tensor(code, data["padding"])
        x = x.view(data["shape"]).to(data["dtype"])
        return x, data["prescale"]

    def compress_binary_tensor(self, binary_tensor):
        assert binary_tensor.dtype == torch.uint8
        assert torch.all((binary_tensor == 0) | (binary_tensor == 1)), "Tensor values must be 0 or 1"

        padding = (8 - binary_tensor.numel() % 8) % 8
        if padding > 0:
            binary_tensor = torch.cat([binary_tensor, torch.zeros(padding, dtype=torch.uint8)])

        binary_tensor = binary_tensor.view(-1, 8)
        packed_tensor = (binary_tensor * self.CONST1).sum(dim=1)
        packed_tensor = packed_tensor.to(torch.uint8)
        return packed_tensor, padding


    def decompress_binary_tensor(self, packed_tensor, padding):
        binary_tensor = ((packed_tensor.unsqueeze(1) & self.CONST2) > 0).to(torch.uint8)
        binary_tensor = binary_tensor.view(-1)[:binary_tensor.numel() - padding]
        return binary_tensor


class KeyCompressor(nn.Module):
    """
    E2E-CommVQ: End-to-End learnable Key compressor with RoPE-Commutative codebook structure.
    Maintains compatibility with EM-trained codebook format for evaluation scripts.
    """
    def __init__(self, feat_dim, layer_idx, quant_bits=1, config=None, num_residuals=21, group_size=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.layer_idx = layer_idx
        self.quant_bits = quant_bits
        self.num_residuals = num_residuals
        self.group_size = group_size
        self.num_groups = feat_dim // group_size  # 1024 // 128 = 8 groups
        
        # Codebook parameters
        # For 12-bit (6+6 bits): 64x64 = 4096 codebook entries per residual
        self.n_bits_half = quant_bits * 3  # 1-bit: 3, 2-bit: 6
        self.codebook_size_half = 2 ** self.n_bits_half  # 1-bit: 8, 2-bit: 64
        self.codebook_size = self.codebook_size_half * self.codebook_size_half  # 1-bit: 64, 2-bit: 4096
        
        # Learnable codebook parameters: theta = [x, y] for each codebook entry in each residual
        # Shape: [num_groups, num_residuals, codebook_size_half, group_size//2, 2]
        # This represents the RoPE-Commutative 2x2 rotation matrix parameters
        self.codebook_params = nn.Parameter(
            torch.randn(self.num_groups, self.num_residuals, self.codebook_size_half, self.group_size // 2, 2) * 0.01
        )
        
        # Learnable encoder network (similar to ValueCompressor)
        self.eps = 1e-6
        self.learnable_scale = nn.Parameter(torch.ones(feat_dim))
        self.learnable_scale.requires_grad = False  # freeze scale initially
        
        # Encoder architecture
        self.norm1 = nn.LayerNorm(feat_dim)
        encoder_dim = self.num_groups * self.num_residuals * self.codebook_size
        self.intermediate_dim = min(4096, encoder_dim)
        self.enc1 = nn.Linear(feat_dim, self.intermediate_dim)
        self.act1 = nn.GELU()
        self.enc2 = nn.Linear(self.intermediate_dim, encoder_dim)
        
        # Build transformation matrix T for RoPE-Commutative structure (same as EM version)
        self._build_transformation_matrix()
        
    def _build_transformation_matrix(self):
        """Build the transformation matrix T that enforces RoPE-Commutative structure."""
        # T: [2*codebook_size, 2*codebook_size_half]
        T = torch.zeros(2 * self.codebook_size, 2 * self.codebook_size_half)
        for A in range(self.codebook_size_half):
            for B in range(self.codebook_size_half):
                idx = A * self.codebook_size_half + B
                T[2*idx, 2*A] = 1
                T[2*idx, 2*B+1] = -1
                T[2*idx+1, 2*B] = 1
                T[2*idx+1, 2*A+1] = 1
        self.register_buffer('T', T.float())
        
    def get_codebook_centers(self):
        """
        Compute codebook centers from learnable parameters using RoPE-Commutative structure.
        This follows the same transformation as EM training: clustering_centers = T @ theta
        
        Returns: [num_groups, num_residuals, codebook_size, group_size]
        """
        # codebook_params: [num_groups, num_residuals, codebook_size_half, group_size//2, 2]
        # where last dim is [x, y] for the RoPE rotation parameters
        
        batch_centers = []
        for g in range(self.num_groups):
            residual_centers = []
            for r in range(self.num_residuals):
                theta_xy = self.codebook_params[g, r]  # [codebook_size_half, group_size//2, 2]
                
                # Convert to theta format: [2*codebook_size_half, group_size//2]
                # Row 2i contains x values, row 2i+1 contains y values
                theta = torch.zeros(2 * self.codebook_size_half, self.group_size // 2, 
                                  dtype=theta_xy.dtype, device=theta_xy.device)
                for i in range(self.codebook_size_half):
                    theta[2*i] = theta_xy[i, :, 0]      # x values
                    theta[2*i+1] = theta_xy[i, :, 1]    # y values
                
                # Apply RoPE-Commutative transformation: centers = T @ theta
                # T: [2*codebook_size, 2*codebook_size_half]
                # theta: [2*codebook_size_half, group_size//2]
                # Result: [2*codebook_size, group_size//2]
                centers = self.T @ theta  # [2*codebook_size, group_size//2]
                
                # Reshape to final format: [codebook_size, group_size]
                # Following EM: .view(CLUSTERING_CENTER_NUM, 2, M_GROUP // 2).permute(0, 2, 1).flatten(1)
                centers = centers.view(self.codebook_size, 2, self.group_size // 2)
                centers = centers.permute(0, 2, 1)  # [codebook_size, group_size//2, 2]
                centers = centers.flatten(1)  # [codebook_size, group_size]
                
                residual_centers.append(centers)
            batch_centers.append(torch.stack(residual_centers, dim=0))
        
        return torch.stack(batch_centers, dim=0)  # [num_groups, num_residuals, codebook_size, group_size]
    
    def encode(self, x, training_method='em'):
        """
        Encode input using either E2E or EM method.
        
        Args:
            x: Input tensor [batch, seq_len, feat_dim]
            training_method: 'e2e' for end-to-end training, 'em' for EM-based inference
            
        Returns:
            If training_method == 'e2e': (quantized_x, prescale, commitment_loss)
            If training_method == 'em': raises NotImplementedError (use evaluation version)
        """
        if training_method == 'em':
            raise NotImplementedError("EM-based encoding should use compress_evaluation.KeyCompressor")
        
        # E2E encoding
        original_shape = x.shape  # [batch, seq_len, feat_dim]
        x = x / (self.learnable_scale + self.eps)
        prescale = torch.norm(x, dim=-1, keepdim=True)
        x = x / (prescale + self.eps)
        
        # Pass through encoder network
        x = self.norm1(x)
        h = self.enc1(x)
        h = self.act1(h)
        logits = self.enc2(h)  # [batch, seq_len, encoder_dim]
        
        # Reshape for residual quantization
        # encoder_dim = num_groups * num_residuals * codebook_size
        logits = logits.view(original_shape[0], original_shape[1], self.num_groups, self.num_residuals, self.codebook_size)
        
        # Get codebook centers
        codebook_centers = self.get_codebook_centers()  # [num_groups, num_residuals, codebook_size, group_size]
        
        # Reshape normalized input for group processing
        # x shape: [batch, seq_len, feat_dim]
        # feat_dim = num_groups * group_size
        x_grouped = x.view(original_shape[0], original_shape[1], self.num_groups, self.group_size)
        
        # Residual quantization with Straight-Through Estimator
        quantized_x = torch.zeros_like(x_grouped)
        residual = x_grouped.clone()
        commitment_loss = 0.0
        
        for r in range(self.num_residuals):
            # Get logits for this residual: [batch, seq_len, num_groups, codebook_size]
            residual_logits = logits[:, :, :, r, :]
            
            # Gumbel-Softmax for differentiable sampling during training
            if self.training:
                # Use Gumbel-Softmax with temperature for differentiable sampling
                indices_soft = torch.nn.functional.gumbel_softmax(residual_logits, tau=1.0, hard=False, dim=-1)
                indices_hard = torch.nn.functional.gumbel_softmax(residual_logits, tau=1.0, hard=True, dim=-1)
                # Straight-through estimator
                indices = indices_hard.detach() + (indices_soft - indices_soft.detach())
            else:
                # During eval, use hard assignment
                indices_hard = torch.nn.functional.one_hot(
                    residual_logits.argmax(dim=-1), num_classes=self.codebook_size
                ).float()
                indices = indices_hard
            
            # Gather quantized vectors: [batch, seq_len, num_groups, group_size]
            # codebook_centers: [num_groups, num_residuals, codebook_size, group_size]
            # indices: [batch, seq_len, num_groups, codebook_size]
            
            # Get codebook centers for this residual: [num_groups, codebook_size, group_size]
            centers_r = codebook_centers[:, r, :, :]  # [G, C, H]
            
            # Use matmul instead of einsum for better memory safety
            # indices: [B, S, G, C], centers_r: [G, C, H]
            # Process each group separately to avoid memory issues
            selected = torch.zeros_like(residual)  # [B, S, G, H]
            for g in range(self.num_groups):
                # indices[:, :, g, :]: [B, S, C]
                # centers_r[g, :, :]: [C, H]
                # result: [B, S, H]
                selected[:, :, g, :] = torch.matmul(indices[:, :, g, :], centers_r[g, :, :])
            
            quantized_x = quantized_x + selected
            residual = residual - selected
            
            # Commitment loss (encourage encoder output to commit to codebook)
            if self.training:
                commitment_loss = commitment_loss + torch.mean(residual ** 2)
        
        commitment_loss = commitment_loss / self.num_residuals
        
        # Restore shape and apply prescale
        quantized_x = quantized_x.view(original_shape)
        quantized_x = quantized_x * (prescale + self.eps)
        quantized_x = quantized_x * (self.learnable_scale + self.eps)
        
        return quantized_x, prescale, commitment_loss
    
    def save_codebook_em_format(self, output_dir):
        """
        Save codebook in the same format as EM training for compatibility with evaluation scripts.
        
        Format: {output_dir}/{layer_idx}_{group_idx}.pt
        Content: {
            "steps": [
                {"mse_loss": float, "theta": tensor, "clustering_centers": tensor},
                ...  # num_residuals times
            ],
            "final_error": float
        }
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current codebook centers
        codebook_centers = self.get_codebook_centers()  # [num_groups, num_residuals, codebook_size, group_size]
        
        for g in range(self.num_groups):
            steps = []
            for r in range(self.num_residuals):
                # Extract theta parameters for this group and residual
                theta = self.codebook_params[g, r].detach().cpu()  # [codebook_size_half, group_size//2, 2]
                
                # Reshape theta to match EM format: [group_size, 2]
                # EM format: theta is [128, 2] where each row is [x_i, y_i]
                theta_reshaped = theta.view(self.codebook_size_half, -1).t()  # [group_size, codebook_size_half] -> wrong shape
                
                # Actually, EM theta format is [2*codebook_size_half, group_size//2]
                # Let's check the EM code again... 
                # From quantize_key_cache.py line 120: theta = A_mat @ B_mat
                # B_mat shape: (m.shape[0], M_GROUP // 2, 2).permute(0, 2, 1).flatten(0, 1)
                # So theta should be [2*codebook_size_half, group_size//2]
                
                theta_for_save = torch.stack([theta[:, :, 0], theta[:, :, 1]], dim=0)  # [2, codebook_size_half, group_size//2]
                theta_for_save = theta_for_save.permute(1, 2, 0).flatten(0, 1)  # [codebook_size_half * group_size//2, 2]
                
                # Actually looking more carefully at line 119-120:
                # B_mat = m.view(m.shape[0], M_GROUP // 2, 2).permute(0, 2, 1).flatten(0, 1)
                # So B_mat is [codebook_size * 2, group_size//2]
                # theta = A_mat @ B_mat gives [2*codebook_size_half, group_size//2]
                
                # Our codebook_params is [codebook_size_half, group_size//2, 2]
                # We need to convert to [2*codebook_size_half, group_size//2] but as [x;y] interleaved
                # Actually the format should be: row 2i = x_i, row 2i+1 = y_i
                theta_em = torch.zeros(2 * self.codebook_size_half, self.group_size // 2)
                for i in range(self.codebook_size_half):
                    theta_em[2*i] = theta[i, :, 0]      # x values
                    theta_em[2*i+1] = theta[i, :, 1]    # y values
                
                # Get clustering centers for this group and residual
                centers = codebook_centers[g, r].detach().cpu()  # [codebook_size, group_size]
                
                # Reshape to match EM format: [codebook_size, 2] 
                # Actually EM saves it as [codebook_size, group_size] directly, but view as (codebook_size, group_size//2, 2)
                # Let me check compress_evaluation.py line 124:
                # clustering_centers = torch.cat([x["clustering_centers"].unsqueeze(0) for x in center_data]).unsqueeze(1)
                # So it's stored as a list form that gets catted, so original is likely [codebook_size, group_size]
                # But needs to be reshaped to work with the 2D structure
                
                # Looking at quantize_key_cache line 121:
                # clustering_centers = (T @ theta).view(CLUSTERING_CENTER_NUM, 2, M_GROUP // 2).permute(0, 2, 1).flatten(1)
                # So it's [codebook_size, group_size] in the final format
                
                # But we need to match the viewing structure for saving
                # The centers need to be viewed as [codebook_size, group_size//2, 2]
                centers_2d = centers.view(self.codebook_size, self.group_size // 2, 2)
                
                # But EM actually saves them flattened back: [codebook_size, group_size]
                # Let me re-check... line 152 shows best_result includes clustering_centers directly
                # And it's converted to list on line 152, so it's saved as is
                
                # I think we should just save in the same structure
                steps.append({
                    "mse_loss": torch.tensor(0.0),  # Placeholder, E2E doesn't compute this the same way
                    "theta": theta_em,  # [2*codebook_size_half, group_size//2]
                    "clustering_centers": centers,  # [codebook_size, group_size]
                })
            
            # Save to file
            filename = os.path.join(output_dir, f"{str(self.layer_idx).zfill(3)}_{g}.pt")
            data = {
                "steps": steps,
                "final_error": 0.0,  # Placeholder
            }
            torch.save(data, filename)
            
        print(f"Saved E2E codebook for layer {self.layer_idx} in EM-compatible format to {output_dir}")
