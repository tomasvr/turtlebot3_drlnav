from collections import deque
import os
import io
import pickle
import socket
import torch

class StorageManager:
    def __init__(self, name, stage, load_session, load_episode, device):
        if load_session and name not in load_session:
            print(f"ERROR: wrong combination of command and model! make sure command is: {name}_agent")
            while True:
                pass
        self.machine_dir = (os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_drl/model/' + str(socket.gethostname()))
        if 'examples' in load_session:
            self.machine_dir = (os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_drl/model/')
        self.name = name
        self.stage = stage
        self.session = load_session
        self.load_episode = load_episode
        self.session_dir = os.path.join(self.machine_dir, self.session)
        self.map_location = device

    def new_session_dir(self):
        i = 0
        session_dir = os.path.join(self.machine_dir, f"{self.name}_{i}")
        while(os.path.exists(session_dir)):
            i += 1
            session_dir = os.path.join(self.machine_dir, f"{self.name}_{i}")
        self.session = f"{self.name}_{i}"
        print(f"making new model dir: {self.session}")
        os.makedirs(session_dir)
        self.session = self.session
        self.session_dir = session_dir

    def delete_file(path):
        if os.path.exists(path):
            os.remove(path)

    # ------------------------------- SAVING -------------------------------

    def network_save_weights(self, network, model_dir, stage, episode):
        filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
        print(f"saving {network.name} model for episode: {episode}")
        torch.save(network.state_dict(), filepath)

    def save_session(self, episode, networks, pickle_data, replay_buffer):
        print(f"saving data for episode: {episode}, location: {self.session_dir}")
        for network in networks:
            self.network_save_weights(network, self.session_dir, self.stage, episode)

        # Store graph data
        with open(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_episode'+str(episode)+'.pkl'), 'wb') as f:
            pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)

        # Store latest buffer (can become very large, multiple gigabytes)
        with open(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_latest_buffer.pkl'), 'wb') as f:
            pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)

        # Delete previous iterations (except every 1000th episode)
        if (episode % 1000 == 0):
            for i in range(episode, episode - 1000, 100):
                for network in networks:
                    self.delete_file(os.path.join(self.session_dir, network.name + '_stage'+str(self.stage)+'_episode'+str(i)+'.pt'))
                self.delete_file(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_episode'+str(i)+'.pkl'))

    def store_model(self, model):
        with open(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_agent.pkl'), 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    # ------------------------------- LOADING -------------------------------

    def network_load_weights(self, network, model_dir, stage, episode):
        filepath = os.path.join(model_dir, str(network.name) + '_stage'+str(stage)+'_episode'+str(episode)+'.pt')
        print(f"loading: {network.name} model from file: {filepath}")
        network.load_state_dict(torch.load(filepath, self.map_location))

    def load_graphdata(self):
        with open(os.path.join(self.session_dir, 'stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.pkl'), 'rb') as f:
            return pickle.load(f)

    def load_replay_buffer(self, size, buffer_path):
        buffer_path = os.path.join(self.machine_dir, buffer_path)
        if (os.path.exists(buffer_path)):
            with open(buffer_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"buffer does not exist: {buffer_path}")
            return deque(maxlen=size)

    def load_model(self):
        model_path = os.path.join(self.session_dir, 'stage'+str(self.stage)+'_agent.pkl')
        try :
            with open(model_path, 'rb') as f:
                return CpuUnpickler(f, self.map_location).load()
        except FileNotFoundError:
            quit(f"The specified model: {model_path} was not found. Check whether you specified the correct stage {self.stage} and model name")

    def load_weights(self, networks):
        for network in networks:
            self.network_load_weights(network, self.session_dir, self.stage, self.load_episode)

class CpuUnpickler(pickle.Unpickler):
    def __init__(self, file, map_location):
        self.map_location = map_location
        super(CpuUnpickler, self).__init__(file)
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.map_location)
        else:
            return super().find_class(module, name)