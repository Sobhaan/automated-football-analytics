import pandas as pd
import numpy as np

def generate_output_df(time, body_position, pressure, turnable, number_of_scans):
    df = pd.DataFrame({
        'Time': time,
        'Body Position': body_position,
        'Pressure': pressure,
        'Turnable': turnable,
        'Number of scans ': number_of_scans
    })
    return df


def update_lists(players, target_id, body_position, pressure, turnable, number_of_scans):
    found = False
    for player in players:
        if player.detection.data['id'] == target_id:
            body_position.append(player.body_orientation)
            pressure.append(player.pressure)
            if pressure == "High Pressure":
                turnable.append("No")
            else:
                turnable.append("Yes")
            number_of_scans.append("None")
            found = True
            break

    if not found:
        body_position.append("None")
        pressure.append("None")
        turnable.append("None")
        number_of_scans.append("None")
    return body_position, pressure, turnable, number_of_scans
