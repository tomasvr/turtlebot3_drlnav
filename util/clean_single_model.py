import numpy
import os
import glob
import sys
import pandas as pd
import socket
import shutil

base_path = os.getenv('DRLNAV_BASE_PATH') + "/src/turtlebot3_drl/model/" + str(socket.gethostname() + "/")

TOP_EPISODES         = 4        # number of best episodes to keep from each model
CLEAN_BUFFER_STORAGE = True     # Delete stored replay buffer states too free up space

def main(args=sys.argv[1:]):
    model = str(args[0])
    session_dir = base_path + model
    PLOT_INTERVAL = 100

    logfile = glob.glob(session_dir + '/_train_*.txt')
    if len(logfile) != 1:
        quit(f"ERROR: found less or more than 1 logfile for: {model}, merge them first (simply copy and paste)!")
    df = pd.read_csv(logfile[0])
    rewards_column = df[' reward']
    rewards = rewards_column.tolist()
    average_rewards = []
    sum_rewards = 0
    episode_range = len(df.index)

    for i in range (episode_range):
        if i % PLOT_INTERVAL == 0 and i > 0:
            average_rewards.append(sum_rewards / PLOT_INTERVAL)
            sum_rewards = 0
        sum_rewards += rewards[i]

    top_episodes = list(numpy.argpartition(numpy.array(average_rewards), -TOP_EPISODES)[-TOP_EPISODES:])
    top_scores = list(numpy.array(average_rewards)[top_episodes])
    top_episodes = [i * PLOT_INTERVAL for i in top_episodes]
    train_stage = logfile[0].split("_train_stage",1)[1][0]
    print(f" cleaning model: {model:<5}, best score: {max(top_scores)}")
    cleanup(session_dir, train_stage, episode_range, top_episodes)

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
        pass

def cleanup(session_dir, stage, end, exclude):
    if CLEAN_BUFFER_STORAGE:
        buffer_files = glob.glob(session_dir + '/buffer_stage_*.pkl')
        for buffer_file in buffer_files:
            delete_file(buffer_file )
    if not os.path.exists(session_dir):
        print(f"model not found! {session_dir}")
        return
    for eps in range(1, end):
        if (not eps in exclude):
            delete_file(os.path.join(session_dir, 'actor' + '_stage'+ stage +'_episode'+str(eps)+'.pt'))
            delete_file(os.path.join(session_dir, 'target_actor' + '_stage'+ stage +'_episode'+str(eps)+'.pt'))
            delete_file(os.path.join(session_dir, 'critic' + '_stage'+ stage +'_episode'+str(eps)+'.pt'))
            delete_file(os.path.join(session_dir, 'target_critic' + '_stage'+ stage +'_episode'+str(eps)+'.pt'))
            delete_file(os.path.join(session_dir, 'stage'+ stage +'_episode'+str(eps)+'.json'))
            delete_file(os.path.join(session_dir, 'stage'+ stage +'_episode'+str(eps)+'.pkl'))


if __name__ == '__main__':
    main()