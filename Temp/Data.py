import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate 5000 data points
n = 5000

# 1. Time of event (00:00 to 15:00 in 15-minute intervals)
start_time = datetime.strptime("00:00", "%H:%M")
end_time = datetime.strptime("15:00", "%H:%M")
time_delta = (end_time - start_time).total_seconds() / 60  # Total minutes

# Generate random minutes within the range
random_minutes = np.random.uniform(0, time_delta, n)
times = [start_time + timedelta(minutes=m) for m in random_minutes]
times = [t.strftime("%H:%M") for t in times]

# 2. Car_ID (1 to n_cars, with some cars appearing multiple times)
n_cars = int(n/3)  # About 1/3 as many cars as data points
car_ids = np.random.randint(1, n_cars+1, n)

# 3. PET (-2 to 6 seconds with 70% positive)
pet_values = np.concatenate([
    np.random.uniform(-2, 0, int(n*0.1)),   # 30% negative values
    np.random.uniform(0, 6, int(n*0.9))     # 70% positive values
    
])
np.random.shuffle(pet_values)  # Shuffle the distribution

# Create DataFrame
df = pd.DataFrame({
    'Time_of_Event': times,
    'Car_ID': car_ids,
    'PET': np.round(pet_values, 2)  # Round to 2 decimal places
})

# Save to CSV
df.to_csv('car_pet_data.csv', index=False)
print("CSV file generated with 5000 data points!")
