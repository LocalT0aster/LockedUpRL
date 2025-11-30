import logging
import re
import matplotlib.pyplot as plt

from rl.logging_utils import DEFAULT_LOG_FILE, setup_logging

LOG_FILE = DEFAULT_LOG_FILE
logger = logging.getLogger(__name__)
setup_logging()

def parse_training(text: str):
    """
    Парсим строки вида:
    [  200/3000] R= -1089.2 avg100=-26701.8 steps= 253 -> Победа Runner
    """
    pattern = re.compile(
        r'\[\s*(\d+)/(?:\d+)\]\s*'
        r'R=\s*([+-]?\d+(?:\.\d+)?)\s*'
        r'avg100=\s*([+-]?\d+(?:\.\d+)?)\s*'
        r'steps=\s*(\d+)',
        re.UNICODE
    )

    episodes = []
    rewards = []
    avg100 = []
    steps = []

    for m in pattern.finditer(text):
        ep = int(m.group(1))
        R = float(m.group(2))
        a100 = float(m.group(3))
        s = int(m.group(4))

        episodes.append(ep)
        rewards.append(R)
        avg100.append(a100)
        steps.append(s)

    return episodes, rewards, avg100, steps


def parse_eval(text: str):
    """
    Парсим строки вида:
    [EVAL  100/300] R_avg=-5148.5 | steps_avg=1170.3 | winC=  1.0% winR= 42.0% draw= 57.0%
    """
    pattern = re.compile(
        r'\[EVAL\s+(\d+)/(?:\d+)\]\s*'
        r'R_avg=([+-]?\d+(?:\.\d+)?)\s*\|\s*'
        r'steps_avg=([+-]?\d+(?:\.\d+)?)\s*\|\s*'
        r'winC=\s*([+-]?\d+(?:\.\d+)?)%\s*'
        r'winR=\s*([+-]?\d+(?:\.\d+)?)%\s*'
        r'draw=\s*([+-]?\d+(?:\.\d+)?)%',
        re.UNICODE
    )

    eval_eps = []
    R_avg = []
    steps_avg = []
    winC = []
    winR = []
    draw = []

    for m in pattern.finditer(text):
        ep = int(m.group(1))
        r_avg = float(m.group(2))
        s_avg = float(m.group(3))
        wc = float(m.group(4))
        wr = float(m.group(5))
        dr = float(m.group(6))

        eval_eps.append(ep)
        R_avg.append(r_avg)
        steps_avg.append(s_avg)
        winC.append(wc)
        winR.append(wr)
        draw.append(dr)

    return eval_eps, R_avg, steps_avg, winC, winR, draw


def main():
    # читаем лог
    with open(LOG_FILE, encoding="utf-8") as f:
        text = f.read()

    # --- TRAINING ---
    tr_eps, tr_R, tr_avg100, tr_steps = parse_training(text)
    logger.info("Parsed training points: %d", len(tr_eps))

    # --- EVAL ---
    ev_eps, ev_Ravg, ev_steps_avg, ev_winC, ev_winR, ev_draw = parse_eval(text)
    logger.info("Parsed eval points: %d", len(ev_eps))

    # ---------- Графики обучения ----------
    # 1) Reward и avg100
    if tr_eps:
        plt.figure()
        plt.plot(tr_eps, tr_R, label="Total reward per episode")
        plt.plot(tr_eps, tr_avg100, label="Moving avg (100 eps)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN training: reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # 2) Кол-во шагов
        plt.figure()
        plt.plot(tr_eps, tr_steps)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("DQN training: episode length (steps)")
        plt.grid(True)
        plt.tight_layout()

    # ---------- Графики оценки (evaluate) ----------
    if ev_eps:
        # 3) Средняя награда
        plt.figure()
        plt.plot(ev_eps, ev_Ravg)
        plt.xlabel("Eval episode index")
        plt.ylabel("Average reward (R_avg)")
        plt.title("DQN evaluation: average reward")
        plt.grid(True)
        plt.tight_layout()

        # 4) Win-rate’ы
        plt.figure()
        plt.plot(ev_eps, ev_winC, label="winC %")
        plt.plot(ev_eps, ev_winR, label="winR %")
        plt.plot(ev_eps, ev_draw, label="draw %")
        plt.xlabel("Eval episode index")
        plt.ylabel("Percent")
        plt.title("DQN evaluation: win/draw rates")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # 5) Средняя длина эпизода в evaluate
        plt.figure()
        plt.plot(ev_eps, ev_steps_avg)
        plt.xlabel("Eval episode index")
        plt.ylabel("Average steps")
        plt.title("DQN evaluation: average episode length")
        plt.grid(True)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
