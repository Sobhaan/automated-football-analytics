# Soccer Video Analytics

A comprehensive computer vision system for analyzing soccer/football match videos. This project provides real-time tracking and analysis of players, ball possession, passes, body orientation, and tactical metrics.

## Features

- **Player Detection & Tracking**: Detect and track players throughout the match using YOLOv8 and Norfair
- **Ball Detection & Tracking**: Track ball movement and possession
- **Team Classification**: Automatically classify players into teams using HSV color analysis
- **Pass Detection**: Identify and log passes between players with frame-accurate timestamps
- **Body Orientation Analysis**: Determine if players are facing towards or away from play
- **Head Orientation & Scanning**: Track head movements to analyze player awareness
- **Pressure Analysis**: Measure defensive pressure on players (High/Medium/Low/None)
- **Interactive Target Selection**: Click to select which player to analyze in detail

## Output Metrics

The system generates a CSV file with the following data for the selected target player:
- **Time**: Timestamp when the player receives a pass
- **Body Position**: Open/Half-Open/Closed orientation
- **Pressure**: Defensive pressure level from opponents
- **Turnable**: Whether the player can turn with the ball
- **Number of Scans**: Head scanning count before receiving the ball

## Installation

### Prerequisites
- Python 3.8.10 
- CUDA-capable GPU (recommended)
- Poetry for dependency management

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/soccer-video-analytics.git
cd soccer-video-analytics
```

2. Install dependencies using Poetry:
```bash
poetry install
```
There will probably be issues with the gluon library, since they utilize a very old numpy version. Make sure to install that library last. Then update the numpy back to 1.23.0.

3. Download model weights:
   - Place your YOLO player detection model in the weights folder (e.g., `weights/players.pt`)
   - Place your ball detection model in the weights folder (e.g., `weights/ball.pt`)
   - The system will automatically download pose estimation models on first run

## Usage

### Basic Usage

```bash
poetry shell
python run.py --video path/to/your/video.mp4
```

### Command Line Arguments

```bash
python run.py [OPTIONS]

Options:
  --video PATH              Path to input video file
  --ball_model PATH         Path to ball detection model (default: weights/ball.engine)
  --player_model PATH       Path to player detection model (default: weights/players.engine)
  --output PATH            Path for output video file (default: output/output.mp4)
  --visualize BOOL         Enable visualization (default: True)
```

### Example

```bash
python run.py \
  --video matches/game1.mp4 \
  --player_model weights/yolov8x.pt \
  --ball_model weights/ball.pt \
  --output results/game1_analyzed.mp4
```

## Workflow

1. **Initialization**: The system loads the video and detection models
2. **Team Detection**: Automatically identifies team colors from the first 60 random frames
3. **Target Selection**: 
   - A window appears showing detected players with IDs
   - Click on a player to select them as the analysis target
   - Press ENTER to confirm or type the ID manually
4. **Processing**: The system tracks all players and analyzes the target player's:
   - Body orientation relative to play direction
   - Defensive pressure from opponents
   - Head scanning behavior
   - Pass reception events
5. **Output**: 
   - Annotated video with visualizations
   - CSV file with timestamped analysis data

## Key Modules

- `run.py`: Main entry point and processing loop
- `auto_id.py`: Interactive player selection interface
- `auto_hsv_generator.py`: Automatic team color detection
- `body_orientation.py`: Player body orientation analysis
- `head_orientation.py`: Head pose and scanning detection
- `soccer/`: Core classes for Player, Ball, Team, Match, and Pass objects
- `inference/`: Detection and classification models
- `run_utils.py`: Helper functions for detection and tracking

## Customization

### Adjusting Detection Thresholds

Edit these values in `run.py`:
```python
player_conf = 0.4  # Player detection confidence
ball_distance_threshold = 55  # Pixels from player to ball for possession
hit_max = 100  # Tracker hit counter maximum
```

### Modifying Team Colors

The system auto-detects team colors, but you can customize in `auto_hsv_generator.py`:
```python
hsv_variance = (15, 60, 60)  # H, S, V variance for color ranges
min_s_filter = 20  # Minimum saturation
min_v_filter = 20  # Minimum value
```

## Output Format

The CSV output contains:
```
Time,Body Position,Pressure,Turnable,Number of scans
12.5,Open,Low Pressure,Yes,3
25.1,Half-Open,High Pressure,No,1
...
```

## Troubleshooting

1. **No players detected**: Lower the `player_conf` threshold
2. **Wrong team classification**: Adjust HSV parameters or manually set team colors
3. **Target player lost**: The system will prompt for reselection
4. **GPU memory issues**: Reduce video resolution or batch size

## Citation

This project builds upon the work from [Tryolabs' blog post on soccer analytics](https://tryolabs.com/blog/2022/10/17/measuring-soccer-ball-possession-ai-video-analytics).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
