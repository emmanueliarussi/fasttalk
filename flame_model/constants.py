import numpy as np
import torch

# Teeth constants
# Upper teeth vertices - representative points for demonstration
width_multiplier = 0.7
height_multiplier = 1.2
TEETH_VERTICES_UPPER = np.array([
    # Front row vertices (main teeth surface)
    [0.0000 * width_multiplier, -0.0400 * height_multiplier, 0.0050],   # Central point (0)
    [-0.0180 * width_multiplier, -0.0400 * height_multiplier, 0.0050],  # Left (1)
    [0.0180 * width_multiplier, -0.0400 * height_multiplier, 0.0050],   # Right (2)
    # Back row vertices
    [0.0000 * width_multiplier, -0.0350 * height_multiplier, 0.0050],   # Central back (3)
    [-0.0180 * width_multiplier, -0.0350 * height_multiplier, 0.0050],  # Left back (4)
    [0.0180 * width_multiplier, -0.0350 * height_multiplier, 0.0050],   # Right back (5)
    # Root vertices
    [0.0000 * width_multiplier, -0.0300 * height_multiplier, 0.0000],   # Root center (6)
    [-0.0180 * width_multiplier, -0.0300 * height_multiplier, 0.0000],  # Root left (7)
    [0.0180 * width_multiplier, -0.0300 * height_multiplier, 0.0000],   # Root right (8)
    
    # Side vertices - Left
    [-0.0300 * width_multiplier, -0.0400 * height_multiplier, -0.001],  # Left front outer (9)
    [-0.0300 * width_multiplier, -0.0350 * height_multiplier, -0.001],  # Left back outer (10)
    [-0.0300 * width_multiplier, -0.0300 * height_multiplier, -0.001],  # Left root outer (11)
    
    # Side vertices - Right
    [0.0300 * width_multiplier, -0.0400 * height_multiplier, -0.001],   # Right front outer (12)
    [0.0300 * width_multiplier, -0.0350 * height_multiplier, -0.001],   # Right back outer (13)
    [0.0300 * width_multiplier, -0.0300 * height_multiplier, -0.001],
])

# Create lower teeth by offsetting the upper teeth down and slightly forward
TEETH_VERTICES_UPPER = TEETH_VERTICES_UPPER + np.array([0.00, -0.012, 0.028])
TEETH_VERTICES_LOWER = TEETH_VERTICES_UPPER + np.array([0.0, 0.012, 0.0])

# Combine upper and lower teeth
TEETH_VERTICES = np.concatenate([TEETH_VERTICES_UPPER, TEETH_VERTICES_LOWER], axis=0)

# Update faces to better connect the wider sides
TEETH_FACES = np.array([
    # Upper teeth faces - main surface
    [0, 1, 3], [1, 4, 3],  # Left central
    [0, 3, 2], [2, 3, 5],  # Right central
    [3, 4, 6], [4, 7, 6],  # Left back
    [3, 6, 5], [5, 6, 8],  # Right back
    
    # Upper teeth faces - enhanced left side panels
    [1, 9, 4],  [9, 10, 4],   # Left outer side front
    [4, 10, 7], [10, 11, 7],  # Left outer side back
    [1, 9, 0],  # Left front corner
    [7, 11, 4], # Left back corner
    
    # Upper teeth faces - enhanced right side panels
    [2, 12, 5], [12, 13, 5],  # Right outer side front
    [5, 13, 8], [13, 14, 8],  # Right outer side back
    [2, 12, 0],  # Right front corner
    [8, 14, 5],  # Right back corner
    
    # Lower teeth faces (offset by 15 for vertex count of upper teeth)
    [15, 16, 18], [16, 19, 18],  # Left central
    [15, 18, 17], [17, 18, 20],  # Right central
    [18, 19, 21], [19, 22, 21],  # Left back
    [18, 21, 20], [20, 21, 23],  # Right back
    
    # Lower teeth faces - enhanced left side panels
    [16, 24, 19], [24, 25, 19],  # Left outer side front
    [19, 25, 22], [25, 26, 22],  # Left outer side back
    [16, 24, 15],  # Left front corner
    [22, 26, 19],  # Left back corner
    
    # Lower teeth faces - enhanced right side panels
    [17, 27, 20], [27, 28, 20],  # Right outer side front
    [20, 28, 23], [28, 29, 23],  # Right outer side back
    [17, 27, 15],  # Right front corner
    [23, 29, 20],  # Right back corner
], dtype=np.int64)

# Update UV coordinates for the wider teeth
TEETH_UVS = np.array([
    # Upper teeth UVs - main surface points
    [0.5, 0.95],  # Central points
    [0.4, 0.95],  # Left points
    [0.6, 0.95],  # Right points
    [0.5, 0.85],  # Back central
    [0.4, 0.85],  # Back left
    [0.6, 0.85],  # Back right
    [0.5, 0.75],  # Root central
    [0.4, 0.75],  # Root left
    [0.6, 0.75],  # Root right
    
    # Upper teeth UVs - side points
    [0.2, 0.95],  # Left front outer
    [0.2, 0.85],  # Left back outer
    [0.2, 0.75],  # Left root outer
    [0.8, 0.95],  # Right front outer
    [0.8, 0.85],  # Right back outer
    [0.8, 0.75],  # Right root outer
    
    # Lower teeth UVs - main surface points
    [0.5, 0.65],  # Central points
    [0.4, 0.65],  # Left points
    [0.6, 0.65],  # Right points
    [0.5, 0.55],  # Back central
    [0.4, 0.55],  # Back left
    [0.6, 0.55],  # Back right
    [0.5, 0.45],  # Root central
    [0.4, 0.45],  # Root left
    [0.6, 0.45],  # Root right
    
    # Lower teeth UVs - side points
    [0.2, 0.65],  # Left front outer
    [0.2, 0.55],  # Left back outer
    [0.2, 0.45],  # Left root outer
    [0.8, 0.65],  # Right front outer
    [0.8, 0.55],  # Right back outer
    [0.8, 0.45],  # Right root outer
])

# Tongue constants
# Base shape for tongue
TONGUE_VERTICES = np.array([
    # Tongue tip
    [0.0000, -0.0100, 0.0200],  # Center
    [-0.0100, -0.0100, 0.0200], # Left
    [0.0100, -0.0100, 0.0200],  # Right
    
    # Tongue middle
    [0.0000, -0.0100, 0.0100],  # Center
    [-0.0120, -0.0100, 0.0100], # Left
    [0.0120, -0.0100, 0.0100],  # Right
    
    # Tongue back
    [0.0000, -0.0080, 0.0000],  # Center
    [-0.0150, -0.0080, 0.0000], # Left
    [0.0150, -0.0080, 0.0000],  # Right
])

# Tongue faces
TONGUE_FACES = np.array([
    [0, 1, 3], [1, 4, 3], # Left front
    [0, 3, 2], [2, 3, 5], # Right front
    [3, 4, 6], [4, 7, 6], # Left back
    [3, 6, 5], [5, 6, 8], # Right back
], dtype=np.int64)

# Tongue UV coordinates
TONGUE_UVS = np.array([
    [0.5, 0.9], # Tip center
    [0.4, 0.9], # Tip left
    [0.6, 0.9], # Tip right
    [0.5, 0.7], # Middle center
    [0.4, 0.7], # Middle left
    [0.6, 0.7], # Middle right
    [0.5, 0.5], # Back center
    [0.4, 0.5], # Back left
    [0.6, 0.5], # Back right
])

# Torso constants
def create_torso_vertices(height=10, radius=8, segments=16):
    vertices = []
    for i in range(height):
        y = -i * 0.02  # Moving down from neck
        for j in range(segments):
            angle = 2 * np.pi * j / segments
            x = radius * 0.015 * np.cos(angle)
            z = radius * 0.015 * np.sin(angle)
            # Add some natural body shape variation
            r_scale = 1.0 + 0.1 * (1 - i/height)  # Slightly wider at top
            x *= r_scale
            z *= r_scale
            vertices.append([x, y, z])
    return np.array(vertices)

TORSO_VERTICES = create_torso_vertices()

# Create faces for torso connecting the rings of vertices
def create_torso_faces(height=10, segments=16):
    faces = []
    for i in range(height-1):
        for j in range(segments):
            v0 = i * segments + j
            v1 = v0 + 1 if j < segments-1 else i * segments
            v2 = v0 + segments
            v3 = v1 + segments
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return np.array(faces, dtype=np.int64)

TORSO_FACES = create_torso_faces()

# Create UV coordinates for torso
def create_torso_uvs(height=10, segments=16):
    uvs = []
    for i in range(height):
        for j in range(segments):
            u = j / (segments - 1)
            v = i / (height - 1)
            uvs.append([u, v])
    return np.array(uvs)

TORSO_UVS = create_torso_uvs()

# Convert all numpy arrays to torch tensors
def numpy_to_torch(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype)

# Create torch tensor versions
TEETH_VERTICES = numpy_to_torch(TEETH_VERTICES)
TEETH_FACES = numpy_to_torch(TEETH_FACES, dtype=torch.long)
TEETH_UVS = numpy_to_torch(TEETH_UVS)

TONGUE_VERTICES = numpy_to_torch(TONGUE_VERTICES)
TONGUE_FACES = numpy_to_torch(TONGUE_FACES, dtype=torch.long)
TONGUE_UVS = numpy_to_torch(TONGUE_UVS)

TORSO_VERTICES = numpy_to_torch(TORSO_VERTICES)
TORSO_FACES = numpy_to_torch(TORSO_FACES, dtype=torch.long)
TORSO_UVS = numpy_to_torch(TORSO_UVS)