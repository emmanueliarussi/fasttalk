''"""
FLAME Model implementation with extended features including teeth, tongue, mouth interior, and torso.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from .constants import (
    TEETH_VERTICES, TEETH_VERTICES_UPPER, TEETH_VERTICES_LOWER,TEETH_FACES, TEETH_UVS,
    TONGUE_VERTICES, TONGUE_FACES, TONGUE_UVS,
    TORSO_VERTICES, TORSO_FACES, TORSO_UVS
)
from .lbs import lbs, batch_rodrigues, vertices2landmarks, blend_shapes

class FLAMEModel(nn.Module):
    """
    Given FLAME parameters, generates a differentiable FLAME function
    which outputs a mesh and 2D/3D facial landmarks.
    """
    def __init__(
        self, 
        n_shape, 
        n_exp, 
        scale=1.0, 
        no_lmks=False,
        add_teeth=True,
        add_tongue=True,
        add_mouth_interior=True,
        add_torso=False,
        add_eyelids=True
    ):
        self.n_shape = n_shape
        self.n_exp = n_exp
        super().__init__()
        self.scale = scale
        self.no_lmks = no_lmks
        
        # Load FLAME model data
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        self.flame_path = os.path.join(_abs_path, 'assets')
        self.flame_ckpt = torch.load(
            os.path.join(self.flame_path, 'FLAME_with_eye.pt'), 
            map_location='cpu', 
            weights_only=True
        )
        flame_model = self.flame_ckpt['flame_model']
        flame_lmk = self.flame_ckpt['lmk_embeddings']
        
        # Set data type
        self.dtype = torch.float32
        
        # Register basic FLAME components
        self.register_buffer('faces_tensor', flame_model['f'])
        self.register_buffer('v_template', flame_model['v_template'])
        
        # Shape components
        shapedirs = flame_model['shapedirs']
        self.register_buffer('shapedirs', torch.cat(
            [shapedirs[:, :, :n_shape], shapedirs[:, :, 300:300 + n_exp]], 2
        ))
        
        # Pose components
        num_pose_basis = flame_model['posedirs'].shape[-1]
        self.register_buffer('posedirs', 
            flame_model['posedirs'].reshape(-1, num_pose_basis).T
        )
        
        # Register other core components
        self.register_buffer('J_regressor', flame_model['J_regressor'])
        parents = flame_model['kintree_table'][0]
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', flame_model['weights'])
        
        # Eye and neck pose defaults
        self.register_buffer('eye_pose', torch.zeros([1, 6], dtype=torch.float32))
        self.register_buffer('neck_pose', torch.zeros([1, 3], dtype=torch.float32))

        # Landmark components - ensure long tensor types
        self.register_buffer('lmk_faces_idx', 
            flame_lmk['static_lmk_faces_idx'].to(dtype=torch.long)
        )
        self.register_buffer('lmk_bary_coords', 
            flame_lmk['static_lmk_bary_coords'].to(dtype=self.dtype)
        )
        self.register_buffer('dynamic_lmk_faces_idx',
            flame_lmk['dynamic_lmk_faces_idx'].to(dtype=torch.long)
        )
        self.register_buffer('dynamic_lmk_bary_coords',
            flame_lmk['dynamic_lmk_bary_coords'].to(dtype=self.dtype)
        )
        self.register_buffer('full_lmk_faces_idx',
            flame_lmk['full_lmk_faces_idx_with_eye'].to(dtype=torch.long)
        )
        self.register_buffer('full_lmk_bary_coords',
            flame_lmk['full_lmk_bary_coords_with_eye'].to(dtype=self.dtype)
        )

        # Setup neck kinematic chain
        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

        # Add additional features if requested
            
        if add_tongue:
            print("Adding tongue to the FLAME model")
            self.add_tongue()
        
        if add_torso:
            print("Adding torso to the FLAME model")
            self.add_torso()
        
        if add_mouth_interior:
            print("Adding mouth interior to the FLAME model")
            self.add_mouth_interior()
        if add_teeth:
            print("Adding teeth to the FLAME model")
            self.add_teeth()
        
        
        if add_eyelids:
            print("Adding eyelids to the FLAME model")
            self.register_buffer(
                'l_eyelid', 
                torch.from_numpy(np.load(f'{os.path.dirname(__file__)}/assets/l_eyelid.npy')).to(self.dtype)[None]
            )
            self.register_buffer(
                'r_eyelid', 
                torch.from_numpy(np.load(f'{os.path.dirname(__file__)}/assets/r_eyelid.npy')).to(self.dtype)[None]
            )
            
    def update_attributes_for_new_verts(self, num_upper_verts=0, num_lower_verts=0, is_torso=False, is_mouth_interior=False):
        """Helper function to update model attributes when adding new vertices"""
        num_new_verts = num_upper_verts + num_lower_verts
        num_verts_orig = self.v_template.shape[0] - num_new_verts
        
        # Update shapedirs
        if not hasattr(self, 'vid_teeth_upper'):  # Only update if not already handled
            self.shapedirs = torch.cat([
                self.shapedirs, 
                torch.zeros_like(self.shapedirs[:num_new_verts])
            ], dim=0)
        
        # Update posedirs
        posedirs = self.posedirs.reshape(len(self.parents)-1, 9, num_verts_orig, 3)
        posedirs = torch.cat([
            posedirs, 
            torch.zeros_like(posedirs[:, :, :num_new_verts])
        ], dim=2)
        self.posedirs = posedirs.reshape(
            (len(self.parents)-1)*9, 
            (num_verts_orig+num_new_verts)*3
        )
        
        # Update J_regressor
        self.J_regressor = torch.cat([
            self.J_regressor, 
            torch.zeros_like(self.J_regressor[:, :num_new_verts])
        ], dim=1)
        
        # Update lbs_weights
        if not hasattr(self, 'vid_teeth_upper'):  # Only update if not already handled for teeth
            self.lbs_weights = torch.cat([
                self.lbs_weights, 
                torch.zeros((num_new_verts, self.lbs_weights.shape[1]), 
                           device=self.lbs_weights.device,
                           dtype=self.lbs_weights.dtype)
            ], dim=0)
            
            if is_torso:
                self.lbs_weights[-num_new_verts:, 0] = 1.0  # Root joint
            elif is_mouth_interior:
                self.lbs_weights[-num_new_verts:, 2] = 1.0  # Jaw joint
            else:
                # Default case
                print("Setting default weights for new vertices")
                self.lbs_weights[-num_new_verts:, 2] = 1.0
    

    def compute_joint_transforms(self):
        # Compute global transforms in order: root -> neck -> jaw
        transforms = []
        for i, parent in enumerate(self.parents):
            if parent == -1:
                transforms.append(self.get_transform(i))
            else:
                transforms.append(torch.matmul(transforms[parent], self.get_transform(i)))
        return transforms

    def add_teeth(self):
        """Add anatomically positioned teeth geometry based on lip positions."""
        # Get reference vertices from lips
        # Note: You'll need to add these lip indices to your constants or mask system
        upper_lip_indices = torch.tensor([1713, 1715, 1716, 1735, 1696, 1694, 1657, 3543, 
                                        2774, 2811, 2813, 2850, 2833, 2832, 2830], 
                                        device=self.v_template.device)
        lower_lip_indices = torch.tensor([1576, 1577, 1773, 1774, 1795, 1802, 1865, 3503,
                                        2948, 2905, 2898, 2881, 2880, 2713, 2712],
                                        device=self.v_template.device)

        # Get lip vertices
        v_lip_upper = self.v_template[upper_lip_indices]
        v_lip_lower = self.v_template[lower_lip_indices]

        # Calculate teeth positions
        mean_dist = (v_lip_upper - v_lip_lower).norm(dim=-1, keepdim=True).mean()
        v_teeth_middle = (v_lip_upper + v_lip_lower) / 2
        v_teeth_middle[:, 1] = v_teeth_middle[:, [1]].mean(dim=0, keepdim=True)
        v_teeth_middle[:, 2] -= mean_dist * 3

        upper_teeth_vertical_offset = -mean_dist * 1.0 # Negative to move down, adjust this value
        v_teeth_upper_edge = v_teeth_middle.clone() + torch.tensor([[0, mean_dist, 0]], device=self.v_template.device)*0.1 + torch.tensor([[0, upper_teeth_vertical_offset, 0]], device=self.v_template.device)
        v_teeth_upper_root = v_teeth_upper_edge + torch.tensor([[0, mean_dist, 0]], device=self.v_template.device) * 2


        # Lower teeth - adjust the vertical offset
        # Try making the offset more negative to move teeth up, based on observation
        lower_teeth_vertical_offset = -mean_dist * 0.0  # Adjust this value to move lower teeth up/down
        v_teeth_lower_edge = v_teeth_middle.clone() - torch.tensor([[0, mean_dist*0.5, 0]], device=self.v_template.device)*0.1 + torch.tensor([[0, lower_teeth_vertical_offset, 0]], device=self.v_template.device)
        # Reduce the multiplier from 6 to 0.5 to bring lower teeth forward (previous successful edit)
        v_teeth_lower_edge -= torch.tensor([[0, 0, mean_dist]], device=self.v_template.device) * 0
        v_teeth_lower_root = v_teeth_lower_edge - torch.tensor([[0, mean_dist, 0]], device=self.v_template.device) * 0.7

        # Add thickness
        thickness = mean_dist * 1.0  # Adjust teeth thickness
        
        # Back vertices
        v_teeth_upper_root_back = v_teeth_upper_root.clone()
        v_teeth_upper_edge_back = v_teeth_upper_edge.clone()
        v_teeth_upper_root_back[:, 2] -= thickness
        v_teeth_upper_edge_back[:, 2] -= thickness

        v_teeth_lower_root_back = v_teeth_lower_root.clone()
        v_teeth_lower_edge_back = v_teeth_lower_edge.clone()
        v_teeth_lower_root_back[:, 2] -= thickness
        v_teeth_lower_edge_back[:, 2] -= thickness

        # Combine all teeth vertices
        num_verts_orig = self.v_template.shape[0]
        v_teeth = torch.cat([
            v_teeth_upper_root,      # num_verts_orig + 0-14 
            v_teeth_lower_root,      # num_verts_orig + 15-29
            v_teeth_upper_edge,      # num_verts_orig + 30-44
            v_teeth_lower_edge,      # num_verts_orig + 45-59
            v_teeth_upper_root_back, # num_verts_orig + 60-74
            v_teeth_upper_edge_back, # num_verts_orig + 75-89
            v_teeth_lower_root_back, # num_verts_orig + 90-104
            v_teeth_lower_edge_back, # num_verts_orig + 105-119
        ], dim=0)
        
        num_verts_teeth = v_teeth.shape[0]
        self.v_template = torch.cat([self.v_template, v_teeth], dim=0)

        # Create vertex indices for different parts
        vid_teeth_upper_root = torch.arange(0, 15, device=self.v_template.device) + num_verts_orig
        vid_teeth_lower_root = torch.arange(15, 30, device=self.v_template.device) + num_verts_orig
        vid_teeth_upper_edge = torch.arange(30, 45, device=self.v_template.device) + num_verts_orig
        vid_teeth_lower_edge = torch.arange(45, 60, device=self.v_template.device) + num_verts_orig
        vid_teeth_upper_root_back = torch.arange(60, 75, device=self.v_template.device) + num_verts_orig
        vid_teeth_upper_edge_back = torch.arange(75, 90, device=self.v_template.device) + num_verts_orig
        vid_teeth_lower_root_back = torch.arange(90, 105, device=self.v_template.device) + num_verts_orig
        vid_teeth_lower_edge_back = torch.arange(105, 120, device=self.v_template.device) + num_verts_orig

        # Combine indices
        vid_teeth_upper = torch.cat([
            vid_teeth_upper_root,
            vid_teeth_upper_edge,
            vid_teeth_upper_root_back,
            vid_teeth_upper_edge_back
        ], dim=0)
        
        vid_teeth_lower = torch.cat([
            vid_teeth_lower_root,
            vid_teeth_lower_edge,
            vid_teeth_lower_root_back,
            vid_teeth_lower_edge_back
        ], dim=0)
        
        # Store teeth vertex indices first - make sure these are contiguous ranges
        self.register_buffer('vid_teeth_upper', vid_teeth_upper)
        self.register_buffer('vid_teeth_lower', vid_teeth_lower)
        
        #set the lbs weights manually
        self.lbs_weights = torch.cat([self.lbs_weights, torch.zeros_like(self.lbs_weights[:num_verts_teeth])], dim=0)  # (V, 5) -> (V+num_verts_teeth, 5)
        self.lbs_weights[vid_teeth_upper, 1] += 1  # move with neck
        self.lbs_weights[vid_teeth_lower, 2] += 1  # move with jaw        
        
        print("LBS weights shape after correct concatenation:", self.lbs_weights.shape)
        
        # Verify weights for last few vertices
        print("Weights for last few vertices:", self.lbs_weights[-5:])
        
        # Store number of teeth vertices
        self.register_buffer('num_teeth_vertices', torch.tensor(num_verts_teeth))
        
        # Update other attributes
        self.update_attributes_for_new_verts(len(vid_teeth_upper), len(vid_teeth_lower))
        
        # Copy shape directions from lips for more natural deformation
        shape_dirs_mean = (self.shapedirs[upper_lip_indices, :, :self.n_shape] + 
                          self.shapedirs[lower_lip_indices, :, :self.n_shape]) / 2
        
        # Expand shapedirs tensor to accommodate new vertices
        shapedirs_expansion = torch.zeros(
            (num_verts_teeth, self.shapedirs.shape[1], self.shapedirs.shape[2]),
            device=self.shapedirs.device,
            dtype=self.shapedirs.dtype
        )
        self.shapedirs = torch.cat([self.shapedirs, shapedirs_expansion], dim=0)
        
        # Now apply shape directions to all teeth vertices
        for vid in [vid_teeth_upper_root, vid_teeth_lower_root, vid_teeth_upper_edge, 
                    vid_teeth_lower_edge, vid_teeth_upper_root_back, vid_teeth_upper_edge_back,
                    vid_teeth_lower_root_back, vid_teeth_lower_edge_back]:
            self.shapedirs[vid, :, :self.n_shape] = shape_dirs_mean

        # Create UV coordinates
        u = torch.linspace(0.62, 0.38, 15, device=self.v_template.device)
        v = torch.linspace(1-0.0083, 1-0.0425, 7, device=self.v_template.device)
        v = v[[3, 2, 0, 1, 3, 4, 6, 5]]
        uv = torch.stack(torch.meshgrid(u, v, indexing='ij'), dim=-1).permute(1, 0, 2).reshape(num_verts_teeth, 2)
        
        # Add UV coordinates
        if hasattr(self, 'verts_uvs'):
            num_verts_uv_orig = self.verts_uvs.shape[0]
            self.verts_uvs = torch.cat([self.verts_uvs, uv], dim=0)

        # Add faces for teeth (you'll need to add your face definitions here)
        # add faces for teeth
        f_teeth_upper = torch.tensor([
            [0, 31, 30],  #0
            [0, 1, 31],  #1
            [1, 32, 31],  #2
            [1, 2, 32],  #3
            [2, 33, 32],  #4
            [2, 3, 33],  #5
            [3, 34, 33],  #6
            [3, 4, 34],  #7
            [4, 35, 34],  #8
            [4, 5, 35],  #9
            [5, 36, 35],  #10
            [5, 6, 36],  #11
            [6, 37, 36],  #12
            [6, 7, 37],  #13
            [7, 8, 37],  #14
            [8, 38, 37],  #15
            [8, 9, 38],  #16
            [9, 39, 38],  #17
            [9, 10, 39],  #18
            [10, 40, 39],  #19
            [10, 11, 40],  #20
            [11, 41, 40],  #21
            [11, 12, 41],  #22
            [12, 42, 41],  #23
            [12, 13, 42],  #24
            [13, 43, 42],  #25
            [13, 14, 43],  #26
            [14, 44, 43],  #27
            [60, 75, 76],  # 56
            [60, 76, 61],  # 57
            [61, 76, 77],  # 58
            [61, 77, 62],  # 59
            [62, 77, 78],  # 60
            [62, 78, 63],  # 61
            [63, 78, 79],  # 62
            [63, 79, 64],  # 63
            [64, 79, 80],  # 64
            [64, 80, 65],  # 65
            [65, 80, 81],  # 66
            [65, 81, 66],  # 67
            [66, 81, 82],  # 68
            [66, 82, 67],  # 69
            [67, 82, 68],  # 70
            [68, 82, 83],  # 71
            [68, 83, 69],  # 72
            [69, 83, 84],  # 73
            [69, 84, 70],  # 74
            [70, 84, 85],  # 75
            [70, 85, 71],  # 76
            [71, 85, 86],  # 77
            [71, 86, 72],  # 78
            [72, 86, 87],  # 79
            [72, 87, 73],  # 80
            [73, 87, 88],  # 81
            [73, 88, 74],  # 82
            [74, 88, 89],  # 83
            [75, 30, 76],  # 84
            [76, 30, 31],  # 85
            [76, 31, 77],  # 86
            [77, 31, 32],  # 87
            [77, 32, 78],  # 88
            [78, 32, 33],  # 89
            [78, 33, 79],  # 90
            [79, 33, 34],  # 91
            [79, 34, 80],  # 92
            [80, 34, 35],  # 93
            [80, 35, 81],  # 94
            [81, 35, 36],  # 95
            [81, 36, 82],  # 96
            [82, 36, 37],  # 97
            [82, 37, 38],  # 98
            [82, 38, 83],  # 99
            [83, 38, 39],  # 100
            [83, 39, 84],  # 101
            [84, 39, 40],  # 102
            [84, 40, 85],  # 103
            [85, 40, 41],  # 104
            [85, 41, 86],  # 105
            [86, 41, 42],  # 106
            [86, 42, 87],  # 107
            [87, 42, 43],  # 108
            [87, 43, 88],  # 109
            [88, 43, 44],  # 110
            [88, 44, 89],  # 111
        ])
        f_teeth_lower = torch.tensor([
            [45, 46, 15],  # 28           
            [46, 16, 15],  # 29
            [46, 47, 16],  # 30
            [47, 17, 16],  # 31
            [47, 48, 17],  # 32
            [48, 18, 17],  # 33
            [48, 49, 18],  # 34
            [49, 19, 18],  # 35
            [49, 50, 19],  # 36
            [50, 20, 19],  # 37
            [50, 51, 20],  # 38
            [51, 21, 20],  # 39
            [51, 52, 21],  # 40
            [52, 22, 21],  # 41
            [52, 23, 22],  # 42
            [52, 53, 23],  # 43
            [53, 24, 23],  # 44
            [53, 54, 24],  # 45
            [54, 25, 24],  # 46
            [54, 55, 25],  # 47
            [55, 26, 25],  # 48
            [55, 56, 26],  # 49
            [56, 27, 26],  # 50
            [56, 57, 27],  # 51
            [57, 28, 27],  # 52
            [57, 58, 28],  # 53
            [58, 29, 28],  # 54
            [58, 59, 29],  # 55
            [90, 106, 105],  # 112
            [90, 91, 106],  # 113
            [91, 107, 106],  # 114
            [91, 92, 107],  # 115
            [92, 108, 107],  # 116
            [92, 93, 108],  # 117
            [93, 109, 108],  # 118
            [93, 94, 109],  # 119
            [94, 110, 109],  # 120
            [94, 95, 110],  # 121
            [95, 111, 110],  # 122
            [95, 96, 111],  # 123
            [96, 112, 111],  # 124
            [96, 97, 112],  # 125
            [97, 98, 112],  # 126
            [98, 113, 112],  # 127
            [98, 99, 113],  # 128
            [99, 114, 113],  # 129
            [99, 100, 114],  # 130
            [100, 115, 114],  # 131
            [100, 101, 115],  # 132
            [101, 116, 115],  # 133
            [101, 102, 116],  # 134
            [102, 117, 116],  # 135
            [102, 103, 117],  # 136
            [103, 118, 117],  # 137
            [103, 104, 118],  # 138
            [104, 119, 118],  # 139
            [105, 106, 45],  # 140
            [106, 46, 45],  # 141
            [106, 107, 46],  # 142
            [107, 47, 46],  # 143
            [107, 108, 47],  # 144
            [108, 48, 47],  # 145
            [108, 109, 48],  # 146
            [109, 49, 48],  # 147
            [109, 110, 49],  # 148
            [110, 50, 49],  # 149
            [110, 111, 50],  # 150
            [111, 51, 50],  # 151
            [111, 112, 51],  # 152
            [112, 52, 51],  # 153
            [112, 53, 52],  # 154
            [112, 113, 53],  # 155
            [113, 54, 53],  # 156
            [113, 114, 54],  # 157
            [114, 55, 54],  # 158
            [114, 115, 55],  # 159
            [115, 56, 55],  # 160
            [115, 116, 56],  # 161
            [116, 57, 56],  # 162
            [116, 117, 57],  # 163
            [117, 58, 57],  # 164
            [117, 118, 58],  # 165
            [118, 59, 58],  # 166
            [118, 119, 59],  # 167
        ])
    

        # Update faces tensor
        self.faces_tensor = torch.cat([
            self.faces_tensor,
            f_teeth_upper + num_verts_orig,
            f_teeth_lower + num_verts_orig
        ], dim=0)

        print(f"Added anatomically positioned teeth with {num_verts_teeth} vertices")

        # After creating vertex indices
        print("Upper teeth vertex indices range:", vid_teeth_upper.min().item(), "to", vid_teeth_upper.max().item())
        print("Lower teeth vertex indices range:", vid_teeth_lower.min().item(), "to", vid_teeth_lower.max().item())
        
        # After setting up weights
        print("LBS weights shape before:", self.lbs_weights.shape)
        print("LBS weights shape after:", self.lbs_weights.shape)

        print(f"Number of faces after adding teeth: {self.faces_tensor.shape[0]}")
        
    def adjust_teeth_position(self, vertices, num_verts_teeth):
        """
        Adjust teeth positions based on lip vertex positions after shape/expression deformations.
        Handles the updated 120-vertex teeth layout (15 vertices x 8 groups).
        Adjusts both upper and lower teeth.
        """
        batch_size = vertices.shape[0]
        
        # Define lip vertex indices
        upper_lip_indices = torch.tensor([1713, 1715, 1716, 1735, 1696, 1694, 1657, 3543, 
                                        2774, 2811, 2813, 2850, 2833, 2832, 2830], 
                                        device=vertices.device)
        lower_lip_indices = torch.tensor([1576, 1577, 1773, 1774, 1795, 1802, 1865, 3503,
                                        2948, 2905, 2898, 2881, 2880, 2713, 2712],
                                        device=vertices.device)
        
        # Get base vertices without teeth
        base_verts = vertices[:, :-num_verts_teeth]
        teeth_verts = vertices[:, -num_verts_teeth:]
        
        # Calculate lip positions and centers
        upper_lip_pos = base_verts[:, upper_lip_indices]
        lower_lip_pos = base_verts[:, lower_lip_indices]
        upper_lip_center = upper_lip_pos.mean(dim=1, keepdim=True)
        lower_lip_center = lower_lip_pos.mean(dim=1, keepdim=True)
        
        # Group indices for different teeth parts based on the new 120-vertex layout
        upper_indices = torch.cat([
            torch.arange(0, 15),      # upper_root
            torch.arange(30, 45),     # upper_edge
            torch.arange(60, 75),     # upper_root_back
            torch.arange(75, 90)      # upper_edge_back
        ]).to(vertices.device)
        
        lower_indices = torch.cat([
            torch.arange(15, 30),     # lower_root
            torch.arange(45, 60),     # lower_edge
            torch.arange(90, 105),    # lower_root_back
            torch.arange(105, 120)    # lower_edge_back
        ]).to(vertices.device)
        
        # Get upper and lower teeth groups
        upper_teeth = teeth_verts[:, upper_indices]
        lower_teeth = teeth_verts[:, lower_indices]
        
        # Calculate offsets and adjustments - refined values for more natural positioning
        jaw_height = (lower_lip_center - upper_lip_center).norm(dim=-1, keepdim=True)
        
        # Dynamic offsets based on jaw opening
        base_upper_offset = torch.tensor([0.0, -0.000, -0.011], device=vertices.device)
        base_lower_offset = torch.tensor([0.0, -0.001, -0.011], device=vertices.device)
        
        # Adjust offsets based on jaw height
        upper_teeth_offset = base_upper_offset.view(1, 1, 3)
        lower_teeth_offset = base_lower_offset.view(1, 1, 3)
        
        # Adjust positions while maintaining teeth structure
        upper_teeth_center = upper_teeth.mean(dim=1, keepdim=True)
        lower_teeth_center = lower_teeth.mean(dim=1, keepdim=True)
        
        upper_teeth_adjusted = upper_teeth + (upper_lip_center + upper_teeth_offset) - upper_teeth_center
        lower_teeth_adjusted = lower_teeth + (lower_lip_center + lower_teeth_offset) - lower_teeth_center
        
        # Reconstruct full teeth vertices tensor
        adjusted_teeth = teeth_verts.clone()
        adjusted_teeth[:, upper_indices] = upper_teeth_adjusted
        adjusted_teeth[:, lower_indices] = lower_teeth_adjusted
        
        # Return combined vertices
        return torch.cat([base_verts, adjusted_teeth], dim=1)
    
    def setup_teeth_weights(self):
        """Setup linear blend skinning weights for teeth vertices"""
        num_joints = self.lbs_weights.shape[1]
        
        print(f"Number of upper teeth vertices: {len(self.vid_teeth_upper)}")
        print(f"Number of lower teeth vertices: {len(self.vid_teeth_lower)}")
        
        # Initialize weights tensor for all teeth vertices
        teeth_weights = torch.zeros((120, num_joints), 
                                  dtype=self.dtype, 
                                  device=self.v_template.device)
        
        # Create masks for upper and lower teeth vertices
        upper_indices = torch.cat([
            torch.arange(0, 15),    # upper_root
            torch.arange(30, 45),   # upper_edge
            torch.arange(60, 90)    # upper_root_back and upper_edge_back
        ]).to(self.v_template.device)
        
        lower_indices = torch.cat([
            torch.arange(15, 30),   # lower_root
            torch.arange(45, 60),   # lower_edge
            torch.arange(90, 120)   # lower_root_back and lower_edge_back
        ]).to(self.v_template.device)
        
        # Set weights for upper teeth - attach to neck joint (index 1)
        teeth_weights[upper_indices, 1] = 1.0  # Neck joint
        
        # Set weights for lower teeth - attach to jaw joint (index 2)
        teeth_weights[lower_indices, 1] = 1.0  # Jaw joint
        
        print("Weight sums for first few upper teeth:", teeth_weights[upper_indices[:5]].sum(dim=1))
        print("Weight sums for first few lower teeth:", teeth_weights[lower_indices[:5]].sum(dim=1))
        print("Joint indices with non-zero weights (upper):", torch.nonzero(teeth_weights[upper_indices[0]]))
        print("Joint indices with non-zero weights (lower):", torch.nonzero(teeth_weights[lower_indices[0]]))
        
        return teeth_weights

    def add_tongue(self):
        """Add tongue geometry to the model"""
        num_verts_orig = self.v_template.shape[0]
        num_verts_tongue = TONGUE_VERTICES.shape[0]
        
        # Add vertices
        self.v_template = torch.cat([
            self.v_template, 
            TONGUE_VERTICES.to(self.dtype).to(self.v_template.device)
        ], dim=0)
        
        # Add faces
        self.faces_tensor = torch.cat([
            self.faces_tensor, 
            TONGUE_FACES.to(self.faces_tensor.device) + num_verts_orig
        ], dim=0)
        
        # Update attributes
        self.update_attributes_for_new_verts(num_verts_tongue)
        print(f"Added tongue with {num_verts_tongue} vertices")

    def add_mouth_interior(self):
        """Add mouth interior geometry by connecting upper and lower lip vertices."""
        # Get upper and lower lip vertices indices
        upper_lip_indices = torch.tensor([1713, 1715, 1716, 1735, 1696, 1694, 1657, 3543, 2774, 2811, 2813, 2850, 2833, 2832, 2830], device=self.v_template.device)
        lower_lip_indices = torch.tensor([1576, 1577, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880, 2713, 2712], device=self.v_template.device)
        
        num_lip_points = len(upper_lip_indices)
        assert num_lip_points == len(lower_lip_indices), "Upper and lower lip index lists must have the same length."

        # Number of subdivisions between upper and lower lip
        h_subdivs = 2
        inward_offset = 0.03 # How much to push the interior vertices inwards (negative z)

        num_verts_orig = self.v_template.shape[0]
        new_verts_list = []
        upper_interior_v_indices = [] # Store indices relative to the start of new vertices
        lower_interior_v_indices = [] # Store indices relative to the start of new vertices
        
        # Create new vertices by interpolating between corresponding lip points
        for i in range(num_lip_points):
            up_v = self.v_template[upper_lip_indices[i]]
            down_v = self.v_template[lower_lip_indices[i]]
            for j in range(h_subdivs):
                t = (j + 1) / (h_subdivs + 1) # Interpolation factor (0 to 1)
                new_point = up_v * (1 - t) + down_v * t
                new_point[2] -= inward_offset # Push inwards slightly
                new_verts_list.append(new_point)
                
                current_new_vert_idx = len(new_verts_list) - 1
                # Assign to upper or lower based on proximity
                if t < 0.5:
                    upper_interior_v_indices.append(current_new_vert_idx)
                else:
                    lower_interior_v_indices.append(current_new_vert_idx)

        if not new_verts_list:
             print("Warning: No vertices generated for mouth interior.")
             return # Avoid errors if no vertices were created

        new_verts_tensor = torch.stack(new_verts_list)
        num_new_verts = new_verts_tensor.shape[0]
        
        # Add new vertices to the template
        self.v_template = torch.cat([self.v_template, new_verts_tensor], dim=0)

        # --- Create faces ---
        new_faces = []
        
        # Helper function to get vertex index (original lip or new vertex)
        def get_vertex_index(lip_point_idx, row_idx):
            if row_idx == 0: # Upper lip row
                return upper_lip_indices[lip_point_idx]
            elif row_idx == h_subdivs + 1: # Lower lip row
                return lower_lip_indices[lip_point_idx]
            else: # Intermediate rows (new vertices)
                # Calculate index within the new_verts block
                new_vert_linear_idx = lip_point_idx * h_subdivs + (row_idx - 1)
                return num_verts_orig + new_vert_linear_idx

        # Iterate through the grid cells to create faces
        for i in range(num_lip_points - 1): # Iterate along the lip connection
            for j in range(h_subdivs + 1): # Iterate vertically across rows (including lips)
                # Get indices of the four corners of the quad
                idx00 = get_vertex_index(i, j)
                idx10 = get_vertex_index(i + 1, j)
                idx01 = get_vertex_index(i, j + 1)
                idx11 = get_vertex_index(i + 1, j + 1)

                # Create two triangles for the quad (ensure consistent winding)
                new_faces.append([idx00, idx10, idx11])
                new_faces.append([idx00, idx11, idx01])

        # Add new faces to the model's faces tensor
        if new_faces:
            self.faces_tensor = torch.cat([
                self.faces_tensor,
                torch.tensor(new_faces, dtype=torch.long, device=self.faces_tensor.device)
            ], dim=0)

        # Update LBS weights and other attributes for the new vertices
        self.update_attributes_for_new_verts(
            num_upper_verts=len(upper_interior_v_indices),
            num_lower_verts=len(lower_interior_v_indices),
            is_mouth_interior=True
        )
        print(f"Added simplified mouth interior with {num_new_verts} vertices ({len(upper_interior_v_indices)} upper, {len(lower_interior_v_indices)} lower) and {len(new_faces)} faces.")

    def add_torso(self):
        """Add torso geometry with a bell/gaussian shape transition from neck"""
        # Define neck base vertices in clockwise order
        neck_base_indices = torch.tensor([
            3260, 3248, 3359, 3360, 3329, 3330, 3372, 3371, 3327,  # Bottom left quadrant
            3322, 3321, 3355, 3354, 3356, 3357,                    # Left quadrant
            3370, 3285, 3289, 3258, 3257, 3255,                    # Top quadrant
            3254, 3273, 3274, 3229, 3228, 3261                     # Right quadrant
        ], device=self.v_template.device)
        
        # If teeth were added, adjust indices
        if hasattr(self, 'num_teeth_vertices'):
            neck_base_indices = neck_base_indices + self.num_teeth_vertices
        
        neck_verts = self.v_template[neck_base_indices]
        neck_center = neck_verts.mean(dim=0)
        
        # Parameters for bell-shaped torso
        torso_height = 0.1
        torso_radius = 0.15
        torso_depth_scale = 0.15  # Scale factor for depth (z-axis)
        torso_offset = torch.tensor([0.02, 0.05, -0.05], device=neck_verts.device)
        
        num_rows = 8
        verts_per_row = len(neck_base_indices)
        
        new_verts = []
        for i in range(num_rows):
            t = i / (num_rows - 1)
            y_offset = -torso_height * t
            
            for j, v in enumerate(neck_verts):
                angle = 2 * np.pi * j / verts_per_row
                
                # Create a smoother transition from neck to torso
                transition = 1 - np.exp(-3 * t)

                # Calculate new position
                x = neck_center[0] + (torso_radius * 1.333) * np.cos(angle) * transition
                z = neck_center[2] + (torso_radius * torso_depth_scale) * np.sin(angle) * transition  # Apply depth scale
                y = neck_center[1] + y_offset
                
                new_v = torch.tensor([x, y, z], device=neck_verts.device)
                new_v += torso_offset
                
                new_verts.append(new_v)
        
        new_verts = torch.stack(new_verts)
        num_new_verts = len(new_verts)
        
        # Add vertices to template
        base_vertex_count = self.v_template.shape[0]
        self.v_template = torch.cat([self.v_template, new_verts], dim=0)
        
        # Create faces for the torso
        faces = []
        for i in range(num_rows - 1):
            for j in range(verts_per_row):
                v1 = base_vertex_count + i * verts_per_row + j
                v2 = base_vertex_count + i * verts_per_row + (j + 1) % verts_per_row
                v3 = base_vertex_count + (i + 1) * verts_per_row + j
                v4 = base_vertex_count + (i + 1) * verts_per_row + (j + 1) % verts_per_row
                
                faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        self.faces_tensor = torch.cat([
            self.faces_tensor,
            torch.tensor(faces, dtype=torch.long, device=self.faces_tensor.device)
        ], dim=0)
        
        # Update attributes
        self.update_attributes_for_torso(num_new_verts)
        
        print(f"Added bell-shaped torso with {num_new_verts} vertices and {len(faces)} faces")
    

    def update_attributes_for_torso(self, num_new_verts):
        """Update attributes specifically for torso vertices"""
        # Update shapedirs
        if hasattr(self, 'shapedirs'):
            shapedirs_expansion = torch.zeros(
                (num_new_verts, self.shapedirs.shape[1], self.shapedirs.shape[2]),
                device=self.shapedirs.device,
                dtype=self.shapedirs.dtype
            )
            self.shapedirs = torch.cat([self.shapedirs, shapedirs_expansion], dim=0)
        
        # Update posedirs
        if hasattr(self, 'posedirs'):
            num_pose_basis = self.posedirs.shape[0]
            posedirs_expansion = torch.zeros(
                (num_pose_basis, num_new_verts * 3),
                device=self.posedirs.device,
                dtype=self.posedirs.dtype
            )
            self.posedirs = torch.cat([self.posedirs, posedirs_expansion], dim=1)
        
        # Update J_regressor
        if hasattr(self, 'J_regressor'):
            j_reg_expansion = torch.zeros(
                (self.J_regressor.shape[0], num_new_verts),
                device=self.J_regressor.device,
                dtype=self.J_regressor.dtype
            )
            self.J_regressor = torch.cat([self.J_regressor, j_reg_expansion], dim=1)
        
        # Update weights for torso
        if hasattr(self, 'lbs_weights'):
            weights_expansion = torch.zeros(
                (num_new_verts, self.lbs_weights.shape[1]),
                device=self.lbs_weights.device,
                dtype=self.lbs_weights.dtype
            )
            weights_expansion[:, 0] = 1.0  # Attach torso to root joint
            self.lbs_weights = torch.cat([self.lbs_weights, weights_expansion], dim=0)
        
        print(f"Updated attributes for torso:")
        print(f"v_template: {self.v_template.shape}")
        print(f"shapedirs: {self.shapedirs.shape if hasattr(self, 'shapedirs') else 'None'}")
        print(f"posedirs: {self.posedirs.shape if hasattr(self, 'posedirs') else 'None'}")
        print(f"lbs_weights: {self.lbs_weights.shape if hasattr(self, 'lbs_weights') else 'None'}")

    def _find_dynamic_lmk_idx_and_bcoords(
            self, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords,
            neck_kin_chain, dtype=torch.float32
        ):
        """
        Find dynamic landmarks based on the head pose
        """
        batch_size = pose.shape[0]

        aa_pose = torch.index_select(
            pose.view(batch_size, -1, 3), 1, neck_kin_chain
        )
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype
        ).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                              dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(
            dynamic_lmk_faces_idx, 0, y_rot_angle
        )
        dyn_lmk_b_coords = torch.index_select(
            dynamic_lmk_b_coords, 0, y_rot_angle
        )

        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def forward(
        self, 
        shape_params=None, 
        expression_params=None, 
        pose_params=None, 
        eye_pose_params=None,
        neck_pose_params=None,
        translation=None,
        eye_params=None
    ):
        """
        Forward pass of the FLAME model.
        
        Args:
            shape_params: (B, n_shape) Shape parameters
            expression_params: (B, n_exp) Expression parameters
            pose_params: (B, 6) Pose parameters
            eye_pose_params: (B, 6) Eye pose parameters
            eye_params: (B, 2) Eye blink parameters
            
        Returns:
            vertices: (B, V, 3) Mesh vertices
            landmarks: (B, L, 3) 3D landmarks (if no_lmks=False)
        """
        batch_size = shape_params.shape[0]
        
        # Set default pose parameters if none provided
        if pose_params is None:
            
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        if neck_pose_params is not None:
            neck_pose_params = neck_pose_params.to(self.dtype)
        else:
            neck_pose_params = self.neck_pose.expand(batch_size, -1)

        # Convert inputs to correct dtype
        shape_params = shape_params.to(self.dtype)
        expression_params = expression_params.to(self.dtype)
        pose_params = pose_params.to(self.dtype)
        eye_pose_params = eye_pose_params.to(self.dtype)
        if eye_params is not None:
            eye_params = eye_params.to(self.dtype)

        #print all shapes
        # print(shape_params.shape, expression_params.shape, pose_params.shape, eye_pose_params.shape)

        # Combine shape and expression parameters
        betas = torch.cat([shape_params, expression_params], dim=1)
        
        # Combine all pose parameters
        full_pose = torch.cat([
            pose_params[:, :3],
            neck_pose_params,
            pose_params[:, 3:],
            eye_pose_params
        ], dim=1)

        # Apply linear blend skinning to get final vertices
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, _ = lbs(
            betas, 
            full_pose, 
            template_vertices,
            self.shapedirs, 
            self.posedirs,
            self.J_regressor, 
            self.parents,
            self.lbs_weights, 
            dtype=self.dtype,
            detach_pose_correctives=False
        )
        # if hasattr(self, 'num_teeth_vertices'):
        #     vertices = self.adjust_teeth_position(vertices, self.num_teeth_vertices)

        # Scale vertices
        if eye_params is not None:
            # Padd r/l eyelid vertices to match the number of vertices
            padding = vertices.shape[1] - self.r_eyelid.shape[1]
            self.r_eyelid = torch.cat([self.r_eyelid, torch.zeros(1, padding, 3)], dim=1)
            self.l_eyelid = torch.cat([self.l_eyelid, torch.zeros(1, padding, 3)], dim=1)
            vertices = vertices + self.r_eyelid.expand(batch_size, -1, -1) * eye_params[:, 1:2, None]
            vertices = vertices + self.l_eyelid.expand(batch_size, -1, -1) * eye_params[:, 0:1, None]
            
        if translation is not None:
            vertices = (vertices + translation) * self.scale
        else:
            vertices = vertices * self.scale

        if self.no_lmks:
            return vertices

        # Compute landmarks
        #landmarks3d = vertices2landmarks(
        #    vertices=vertices,
        #    faces=self.faces_tensor.long(),  # Ensure faces are long tensor
        #    lmk_faces_idx=self.full_lmk_faces_idx.long(),  # Ensure landmark faces are long tensor
        #    lmk_bary_coords=self.full_lmk_bary_coords
        #)
        
        # Process landmarks for eyes
        #landmarks3d = reselect_eyes(vertices, landmarks3d)
        
        # Add eyelid blending if enabled
        if hasattr(self, 'l_eyelid') and eye_params is not None:
             left_blend = self.l_eyelid * eye_params[:, 0:1].unsqueeze(-1)
             right_blend = self.r_eyelid * eye_params[:, 1:2].unsqueeze(-1)
             vertices = vertices + left_blend + right_blend

        return vertices
    
    def get_faces(self):
        """Return the faces tensor of the mesh."""
        return self.faces_tensor.long()

    def _vertices2landmarks(self, vertices):
        """Convert vertices to landmarks."""
        landmarks3d = vertices2landmarks(
            vertices, 
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1)
        )
        landmarks3d = reselect_eyes(vertices, landmarks3d)
        return landmarks3d

def rot_mat_to_euler(rot_mats):
    """
    Convert rotation matrices to euler angles.
    
    Args:
        rot_mats: (B, 3, 3) Batch of rotation matrices
        
    Returns:
        (B,) Euler angles in radians
    """
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

def reselect_eyes(vertices, lmks70):
    """
    Recompute eye landmarks for better accuracy.
    
    Args:
        vertices: (B, V, 3) Mesh vertices
        lmks70: (B, 70, 3) Original landmarks
        
    Returns:
        (B, 70, 3) Updated landmarks with corrected eye positions
    """
    lmks70 = lmks70.clone()
    
    # Eye vertex indices in the FLAME mesh
    eye_in_shape = [
        2422, 2422, 2452, 2454, 2471, 3638, 2276, 2360, 3835, 
        1292, 1217, 1146, 1146, 999, 827
    ]
    eye_in_shape_reduce = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
    
    # Get eye vertices
    cur_eye = vertices[:, eye_in_shape]
    
    # Average some positions for smoother results
    cur_eye[:, 0] = (cur_eye[:, 0] + cur_eye[:, 1]) * 0.5
    cur_eye[:, 2] = (cur_eye[:, 2] + cur_eye[:, 3]) * 0.5
    cur_eye[:, 11] = (cur_eye[:, 11] + cur_eye[:, 12]) * 0.5
    
    # Select reduced set of eye vertices
    cur_eye = cur_eye[:, eye_in_shape_reduce]
    
    # Update landmark positions
    lmks70[:, [37, 38, 40, 41, 43, 44, 46, 47]] = cur_eye[:, [1, 2, 4, 5, 7, 8, 10, 11]]
    
    return lmks70

class Struct(object):
    """Simple object to store attributes."""
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
            
    def register_buffer(self, name, tensor):
        """Add method to match PyTorch's register_buffer functionality"""
        setattr(self, name, tensor)

    def update(self, faces_tensor, textures_idx):
        """Update masks after adding new geometry"""
        if hasattr(self, 'v'):
            # Update vertex masks if they exist
            for region_name in dir(self.v):
                if not region_name.startswith('_'):  # Skip private attributes
                    mask = getattr(self.v, region_name)
                    if isinstance(mask, torch.Tensor):
                        # Keep the mask as is since vertex indices are already correct
                        continue

# Optional: Add methods for expression transfer
def transfer_expression(source_expr, target_model):
    """
    Transfer expression parameters from source to target.
    
    Args:
        source_expr: Expression parameters from source
        target_model: Target FLAME model instance
    """
    with torch.no_grad():
        target_model.expression_params.copy_(source_expr)
    return target_model

# Optional: Add methods for pose transfer
def transfer_pose(source_pose, target_model):
    """
    Transfer pose parameters from source to target.
    
    Args:
        source_pose: Pose parameters from source
        target_model: Target FLAME model instance
    """
    with torch.no_grad():
        target_model.pose_params.copy_(source_pose)
    return target_model

if __name__ == '__main__':
    # Example usage
    model = FLAMEModel(
        n_shape=100,
        n_exp=50,
        add_teeth=True,
        add_tongue=True,
        add_mouth_interior=True,
        add_torso=True
    )
    
    # Generate random input parameters
    batch_size = 1
    shape_params = torch.zeros(batch_size, 100)
    expression_params = torch.zeros(batch_size, 50)
    pose_params = torch.zeros(batch_size, 6)
    
    # Forward pass
    vertices, landmarks = model(shape_params, expression_params, pose_params)
    print(f"Generated mesh with {vertices.shape[1]} vertices and {landmarks.shape[1]} landmarks")