import pandas as pd
import numpy as np

def generate_output_df(time, pressure, turnable, number_of_scans):
    df = pd.DataFrame({
        'Time': time,
        'Pressure': pressure,
        'Turnable': turnable,
        'Number of scans ': number_of_scans
    })
    return df


def update_lists(players, target_id, pressure, number_of_scans):
    found = False
    for player in players:
        if player.detection.data['id'] == target_id:
            pressure.append(player.pressure)
            number_of_scans.append(player.scanning)
            found = True
            break

    if not found:
        pressure.append(pressure[-1])
        number_of_scans.append(number_of_scans[-1])
    return pressure, number_of_scans
