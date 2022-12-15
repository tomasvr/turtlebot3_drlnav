import numpy
import os
import glob
import sys
import pandas as pd
import socket
import shutil

from clean_single_model import TOP_EPISODES, CLEAN_BUFFER_STORAGE

base_path = os.getenv('DRLNAV_BASE_PATH') + "/src/turtlebot3_drl/model/" + str(socket.gethostname() + "/")

TOP_EPISODES         = TOP_EPISODES             # number of best episodes to keep from each model
CLEAN_BUFFER_STORAGE = CLEAN_BUFFER_STORAGE     # Delete stored replay buffer states too free up space
CUTOFF_EPISODE_COUNT = 50000                      # Delete models with less than this amount of episode
CUTOFF_REWARD_SCORE  = 1000                     # Delete models that never scored higher reward than this value

def main():
    skipped = []
    dirs = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
    for dir in dirs:
        session_dir = base_path + dir
        PLOT_INTERVAL = 100

        logfile = glob.glob(session_dir + '/_train_*.txt')
        if len(logfile) != 1:
            print(f"ERROR: found less or more than 1 logfile for: {dir}, skipping!")
            skipped.append(session_dir[-8:])
            continue
        df = pd.read_csv(logfile[0])
        rewards_column = df[' reward']
        rewards = rewards_column.tolist()
        average_rewards = []
        sum_rewards = 0
        episode_range = len(df.index)

        if episode_range < CUTOFF_EPISODE_COUNT:
            shutil.rmtree(session_dir)
            print(f"removing model: {dir:<15} too few episodes: {episode_range}")
            continue

        for i in range (episode_range):
            if i % PLOT_INTERVAL == 0 and i > 0:
                average_rewards.append(sum_rewards / PLOT_INTERVAL)
                sum_rewards = 0
            sum_rewards += rewards[i]

        top_episodes = list(numpy.argpartition(numpy.array(average_rewards), -TOP_EPISODES)[-TOP_EPISODES:])
        top_scores = list(numpy.array(average_rewards)[top_episodes])
        top_episodes = [i * PLOT_INTERVAL for i in top_episodes]
        train_stage = logfile[0].split("_train_stage",1)[1][0]
        if max(top_scores) < CUTOFF_REWARD_SCORE:
            shutil.rmtree(session_dir)
            print(f"removing model: {dir:<15} too low score:   {max(top_scores)}")
            continue
        print(f"keeping  model: {dir:<15} with best score: {max(top_scores)}")
        cleanup(session_dir, train_stage, 20000, top_episodes)
    print(f"skipped {len(skipped)} models because of multiple train files: {skipped}")

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
        pass

def cleanup(session_dir, stage, end, exclude):
    if CLEAN_BUFFER_STORAGE:
        buffer_files = glob.glob(session_dir + '/buffer_stage_*.pkl')
        for buffer_file in buffer_files:
            delete_file(buffer_file )
    # Delete previous iterations
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