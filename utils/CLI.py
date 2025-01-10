import argparse
import textwrap
from enum import Enum

class Difficulty(Enum):
    EASY = 1          
    MEDIUM = 2        
    HARD = 3          
    VERY_HARD = 4

    @staticmethod
    def parse(difficulty):
        if difficulty == Difficulty.EASY:
            return False, 1, 10
        if difficulty == Difficulty.MEDIUM:
            return True, 1, 10
        if difficulty == Difficulty.HARD:
            return True, 4, 2
        if difficulty == Difficulty.VERY_HARD:
            return True, 6, 2


def cli():
    parser = argparse.ArgumentParser(
        prog='RobôCIn Software Challenge', 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Challenge difficulty level:
            1 - Easy: Static obstacles, single robot, and 10 targets generated successively.
            2 - Medium: Dynamic obstacles, single robot, and 10 targets generated successively.
            3 - Hard: Dynamic obstacles, robot-to-target assignment increasing successively up to 4, followed by 10 generations of targets after reaching 4 robots.
            4 - Very Hard: Dynamic obstacles, robot-to-target assignment increasing successively up to 6, followed by 10 generations of targets after reaching 6 robots.
        """))
    
    parser.add_argument(
        '-d', 
        '--difficulty', 
        type=int, 
        default=1, 
        help='Difficulties: 1, 2, 3 or 4 / Default = 1')

    return parser.parse_args()