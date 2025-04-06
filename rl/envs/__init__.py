from .drink import DrinkVecEnv, DrinkCallback
from .kettle import KettleVecEnv, KettleCallback
from .stapler import StaplerVecEnv, StaplerCallback
from .sprayer import SprayerVecEnv, SprayerCallback
from .ballpoint import BallpointVecEnv, BallpointCallback
dextog_env_map = {
    "drink": DrinkVecEnv,
    "kettle": KettleVecEnv,
    "stapler": StaplerVecEnv,
    "sprayer": SprayerVecEnv,
    "ballpoint": BallpointVecEnv
}
dextog_callback_map = {
    "drink": DrinkCallback,
    "kettle": KettleCallback,
    "stapler": StaplerCallback,
    "sprayer": SprayerCallback,
    "ballpoint": BallpointCallback
}
