import pandas as pd
import numpy as np

def generate_output_df(time, body_position, pressure, turnable):
    df = pd.DataFrame({
        'Time': time,
        'Body Position': body_position,
        'Pressure': pressure,
        'Turnable': turnable,
    })
    return df


def update_lists(players, target_id, body_position, pressure, turnable):
    found = False
    for player in players:
        if player.detection.data['id'] == target_id:
            body_position.append(player.body_orientation)
            pressure.append(player.pressure)
            if (pressure == "High Pressure" or pressure == "Medium Pressure") and body_position == "Closed":
                turnable.append("No")
            else:
                turnable.append("Yes")
            found = True
            break

    if not found:
        body_position.append(body_position[-1])
        pressure.append(pressure[-1])
        turnable.append(turnable[-1])
    return body_position, pressure, turnable
