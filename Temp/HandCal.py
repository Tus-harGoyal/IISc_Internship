import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

df1 = pd.read_csv(r"D:\T\test_codeEVT\nd\ped_smooth.csv")
df2 = pd.read_csv(r"D:\T\test_codeEVT\nd\moto_smooth.csv")

ped = df1[df1['Track ID'] == 382].copy()
moto= df2[df2['Track ID'] == 399].copy()
# Round to nearest 0.16s (export interval)
EXPORT_INTERVAL = 0.16
ped['TimeStamp_rounded'] = np.round(ped['TimeStamp'] / EXPORT_INTERVAL) * EXPORT_INTERVAL
moto['TimeStamp_rounded'] = np.round(moto['TimeStamp'] / EXPORT_INTERVAL) * EXPORT_INTERVAL

# Update TIME_BUFFER to match (recommend 0.16s or 0.32s - 1-2 export intervals)
TIME_BUFFER = 0.0  # ±1 export frame


# Bounding box dimensions
PED_BOX = (0.3, 0.3)  # (length, width)
moto_BOX = (1.87, 0.64)


# Parameters

DIST_THRESH = 1   # Initial distance threshold

def get_rotated_corners(x, y, heading, length, width):
    """Calculate rotated bounding box corners"""
    half_l = length / 2
    half_w = width / 2
    
    corners = np.array([
        [ half_l,  half_w],  # Front right
        [ half_l, -half_w],  # Front left
        [-half_l, -half_w],  # Rear left
        [-half_l,  half_w]   # Rear right
    ])
    
    rad = np.radians(heading)
    rot_mat = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad),  np.cos(rad)]
    ])
    
    rotated = np.dot(corners, rot_mat.T) + np.array([x, y])
    return rotated

conflicts = []

for _, ped_row in ped.iterrows():
    # Temporal proximity check using rounded timestamps
    time_window = (ped_row['TimeStamp_rounded'] - TIME_BUFFER, 
                   ped_row['TimeStamp_rounded'] + TIME_BUFFER)
    
    moto_matches = moto[
        moto['TimeStamp_rounded'].between(*time_window) & 
        (abs(moto['TimeStamp_rounded'] - ped_row['TimeStamp_rounded']) <= TIME_BUFFER)
    ]
    
    for _, moto_row in moto_matches.iterrows():
        # Center-to-center distance check
        center_dist = np.hypot(moto_row['x_smooth'] - ped_row['x_smooth'],
                               moto_row['y_smooth'] - ped_row['y_smooth'])
        
        if center_dist <= DIST_THRESH:
            # Get precise bounding boxes
            ped_corners = get_rotated_corners(
                ped_row['x_smooth'], ped_row['y_smooth'],
                ped_row['HA'], *PED_BOX
            )
            
            moto_corners = get_rotated_corners(
                moto_row['x_smooth'], moto_row['y_smooth'],
                moto_row['HA'], *moto_BOX
            )
            
            # Minimum corner-to-corner distance
            dist_matrix = cdist(ped_corners, moto_corners)
            min_dist = np.min(dist_matrix)
            
            conflicts.append({
                'Ped_Time': ped_row['TimeStamp'],
                'Ped_Time_Rounded': ped_row['TimeStamp_rounded'],
                'moto_Time': moto_row['TimeStamp'],
                'moto_Time_Rounded': moto_row['TimeStamp_rounded'],
                'Time_Diff': moto_row['TimeStamp_rounded'] - ped_row['TimeStamp_rounded'],
                'Center_Distance': center_dist,
                'Min_Corner_Distance': min_dist,
                'Ped_X': ped_row['x_smooth'],
                'Ped_Y': ped_row['y_smooth'],
                'moto_X': moto_row['x_smooth'],
                'moto_Y': moto_row['y_smooth'],
                'Ped_VX': ped_row['vx_smooth'],
                'Ped_VY': ped_row['vy_smooth'],
                'moto_VX': moto_row['vx_smooth'],
                'moto_VY': moto_row['vy_smooth'],
                'Ped_aX': ped_row['vx_smooth'],
                'Ped_aY': ped_row['vy_smooth'],
                'moto_aX': moto_row['vx_smooth'],
                'moto_aY': moto_row['vy_smooth'],
                'moto_Heading': moto_row['HA'],
                'Ped_Heading': ped_row['HA']
            })

# Create results DataFrame
results = pd.DataFrame(conflicts)

# Remove duplicate conflicts (same rounded timestamps)
results = results.drop_duplicates(
    subset=['Ped_Time_Rounded', 'moto_Time_Rounded'], 
    keep='first'
)
results['ATTC']=((results['Ped_VX']-results['moto_VX'])**2 + (results['Ped_VY']-results['moto_VY'])**2)**0.5+((results['Ped_aX'])**2 + ((results['moto_aY'])**2)**0.5)*0.16
if not results.empty:
    print(f"Found {len(results)} conflict points (±{TIME_BUFFER}s, ≤{DIST_THRESH}m):")
    print(results[['Ped_Time_Rounded', 'moto_Time_Rounded', 
                 'Time_Diff', 'Center_Distance', 'Min_Corner_Distance', 'Ped_VX','Ped_VY','moto_VX','moto_VY', 'Ped_aX','Ped_aY','moto_aX','moto_aY','ATTC']])
                 
else:
    print(f"No conflicts found (±{TIME_BUFFER}s, ≤{DIST_THRESH}m)")
