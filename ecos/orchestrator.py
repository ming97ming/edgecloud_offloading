import random
import numpy as np
import os

from ecos.agent import Agent
from ecos.simulator import Simulator


class Orchestrator:
    def __init__(self, _policy, id):
        self.policy = _policy

        self.training_enable = False
        # RL training
        if self.policy != "RANDOM":
            self.file_path = './ecos_result/model_' + str(id) + "/"
            folder_path = Simulator.get_instance().get_loss_folder_path()
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            self.loss_file = open(folder_path + "/loss_log_" + str(id) + ".txt", 'w')
            # create folder
            self.agent = Agent(Simulator.get_instance().get_num_of_edge() + 1, self.file_path)
            # list for training
            self.state_space = 33
            self.action_space = Simulator.get_instance().get_num_of_edge() + 1
            self.state_list = np.empty((0, self.state_space), dtype=np.float64)
            self.actions_list = np.empty((0, self.action_space), dtype=np.float64)
            self.rewards_list = np.empty((0, 1), dtype=np.float64)
            self.next_states_list = np.empty((0, self.state_space), dtype=np.float64)

            self.state = np.zeros(6)
            self.action = None
            self.reward = 0
            self.cumulative_reward = 0
            self.epoch = 1
            # self.replay = ReplayBuffer(20, Simulator.get_instance().get_num_of_edge())
            self.id = id

    def offloading_target(self, task, source):
        collaborationTarget = 0
        simul = Simulator.get_instance()

        if self.policy == "RANDOM":
            num_of_edge = simul.get_num_of_edge()
            selectServer = random.randrange(1, num_of_edge + 1)
            collaborationTarget = selectServer

        elif self.policy == "TaskClassification":
            if not self.training_enable:
                self.training_enable = True

            if Simulator.get_instance().get_clock() < Simulator.get_instance().get_warmup_time():
                num_of_edge = simul.get_num_of_edge()
                selectServer = random.randrange(1, num_of_edge + 1)
                collaborationTarget =selectServer
                return collaborationTarget

            edge_manager = Simulator.get_instance().get_scenario_factory().get_edge_manager()
            cloud_manager = Simulator.get_instance().get_scenario_factory().get_cloud_manager()
            edge_list = edge_manager.get_node_list()
            edge_resource_list = []
            cloud_list = cloud_manager.get_node_list()
            cloud_resource_list = []
            edge_network_list = []
            cloud_network_list = []
            waiting_task_inedge = []
            task_list = []
            sensitive_task = []
            non_sensitive_task = []

            for edge in edge_list:
                edge_resource_list.append(edge.get_CPU() / 10000 )
                task_list.append(edge.get_waiting_list())
                waiting_task_inedge.append(len(edge.get_waiting_list()))

            for task in task_list:
                if task.get_task_deadline() < 200:
                    sensitive_task.append(task)
                else:
                    non_sensitive_task.append(task)

            for link in edge_manager.get_link_list():
                edge_network_list.append(link.get_delay())

            maxvalue = max(waiting_task_inedge)

            if maxvalue == 0:
                maxvalue = 1

            for i in range(len(waiting_task_inedge)):
                waiting_task_inedge[i] = waiting_task_inedge[i] / maxvalue

            for cloud in cloud_list:
                cloud_resource_list.append(cloud.get_cloud_CPU() / 100000)
                cloud_network_list.append(cloud_manager.get_cloud_network().get_delay())

            state_ = [task.get_remain_size() / 1000] + [task.get_task_deadline() / 1000] + \
                     edge_resource_list + edge_network_list + waiting_task_inedge + cloud_resource_list + cloud_network_list
            state = np.array(state_, ndmin=2)

            # model training
            if self.action is not None:
                self.state_list = np.append(self.state_list[-100000:],
                                                np.array(self.state, ndmin=2), axis=0)
                self.actions_list = np.append(self.actions_list[-100000:],
                                         np.array(self.action, ndmin=2), axis=0)
                self.rewards_list = np.append(self.rewards_list[-100000:],
                                         np.array(self.reward, ndmin=2), axis=0)
                self.next_states_list = np.append(self.next_states_list[-100000:],
                                             np.array(state, ndmin=2), axis=0)

            # edit
            action = self.agent.sample_action(state)
            self.action = np.array(action, ndmin=2)
            self.state = state
            action_sample = np.random.choice(Simulator.get_instance().get_num_of_edge() + 1, p=np.squeeze(action))
            collaborationTarget = action_sample
            buffer_time = 0

            # estimate reward
            # processing time
            if action_sample == 0:
                required_resource = task.get_input_size() / task.get_task_deadline()
                processing_time = required_resource / cloud_resource_list[0]
                transmission_time = cloud_manager.get_cloud_network().get_delay()
                completion_time = processing_time + transmission_time

                if task.get_task_deadline() > completion_time:
                    task_fail = 0
                else:
                    task_fail = 1

            else:
                required_resource = task.get_input_size() / task.get_task_deadline()
                processing_time = required_resource / edge_resource_list[action_sample - 1]
                transmission_time = 0

                link_list = edge_manager.get_link_list()

                if source != action_sample:
                    for link in link_list:
                        link_status = link.get_link()
                        set = [source, action_sample]

                        if sorted(set) == sorted(link_status):
                            transmission_time = link.get_delay()
                            break

                waiting_task_inedge = edge_list[action_sample - 1].get_waiting_list()
                buffer_time = 0

                for task in waiting_task_inedge:
                    buffer_time += task.get_input_size() / task.get_task_deadline()

                completion_time = processing_time + transmission_time + buffer_time

                if task.get_task_deadline() > completion_time:
                    task_fail = 0
                else:
                    task_fail = 1

            for i in sensitive_task:
                self.reward = 1/completion_time

            for i in non_sensitive_task:
                self.reward = task.get_task_deadline() - completion_time

            self.epoch += 1

            print("=======================")
            print("state:", state_)
            print("source:", source, " target:", collaborationTarget, "action:", self.action)
            print("reward:", self.reward, "processing:", processing_time,
                  "transmission:", transmission_time, "waiting:", buffer_time)





        elif self.policy == "A2C":
            if not self.training_enable:
                self.training_enable = True

            if Simulator.get_instance().get_clock() < Simulator.get_instance().get_warmup_time():
                num_of_edge = simul.get_num_of_edge()
                selectServer = random.randrange(1, num_of_edge + 1)
                collaborationTarget = selectServer
                return collaborationTarget

            edge_manager = Simulator.get_instance().get_scenario_factory().get_edge_manager()
            cloud_manager = Simulator.get_instance().get_scenario_factory().get_cloud_manager()
            edge_list = edge_manager.get_node_list()
            edge_resource_list = []
            cloud_list = cloud_manager.get_node_list()
            cloud_resource_list = []
            edge_network_list = []
            cloud_network_list = []
            waiting_task_inedge = []

            for edge in edge_list:
                edge_resource_list.append(edge.get_CPU() / 100000)
                waiting_task_inedge.append(len(edge.get_waiting_list()))

            for link in edge_manager.get_link_list():
                edge_network_list.append(link.get_delay())

            maxvalue = max(waiting_task_inedge)

            if maxvalue == 0:
                maxvalue = 1

            for i in range(len(waiting_task_inedge)):
                waiting_task_inedge[i] = waiting_task_inedge[i]/maxvalue


            for cloud in cloud_list:
                cloud_resource_list.append(cloud.get_cloud_CPU() / 100000)
                cloud_network_list.append(cloud_manager.get_cloud_network().get_delay())

            state_ = [task.get_remain_size() / 1000] + [task.get_task_deadline() / 1000] + \
                     edge_resource_list + edge_network_list + waiting_task_inedge + cloud_resource_list + cloud_network_list
            state = np.array(state_, ndmin=2)

            # model training
            if self.action is not None:
                self.state_list = np.append(self.state_list[-100000:],
                                            np.array(self.state, ndmin=2), axis=0)
                self.actions_list = np.append(self.actions_list[-100000:],
                                              np.array(self.action, ndmin=2), axis=0)
                self.rewards_list = np.append(self.rewards_list[-100000:],
                                              np.array(self.reward, ndmin=2), axis=0)
                self.next_states_list = np.append(self.next_states_list[-100000:],
                                                  np.array(state, ndmin=2), axis=0)

            # edit
            action = self.agent.sample_action(state)
            self.action = np.array(action, ndmin=2)
            self.state = state
            action_sample = np.random.choice(Simulator.get_instance().get_num_of_edge()+1, p=np.squeeze(action))
            collaborationTarget = action_sample
            buffer_time=0

            # estimate reward
            # processing time
            if action_sample == 0:
                required_resource = task.get_input_size()/task.get_task_deadline()
                processing_time = required_resource / cloud_resource_list[0]
                transmission_time = cloud_manager.get_cloud_network().get_delay()
                completion_time = processing_time + transmission_time

                if task.get_task_deadline() > completion_time:
                    task_fail = 0
                else:
                    task_fail = 1

            else:
                required_resource = task.get_input_size()/task.get_task_deadline()
                processing_time = required_resource / edge_resource_list[action_sample-1]
                transmission_time = 0

                link_list = edge_manager.get_link_list()

                if source != action_sample:
                    for link in link_list:
                        link_status = link.get_link()
                        set = [source, action_sample]

                        if sorted(set) == sorted(link_status):
                            transmission_time = link.get_delay()
                            break

                waiting_task_inedge = edge_list[action_sample-1].get_waiting_list()
                buffer_time = 0

                for task in waiting_task_inedge:
                    buffer_time += task.get_input_size()/task.get_task_deadline()

                completion_time = processing_time + transmission_time + buffer_time

                if task.get_task_deadline() > completion_time:
                    task_fail = 0
                else:
                    task_fail = 1

            self.reward = -1 * (completion_time + task_fail)
            self.epoch += 1

            print("=======================")
            print("state:", state_)
            print("source:", source, " target:", collaborationTarget, "action:", self.action)
            print("reward:", self.reward, "processing:", processing_time,
                  "transmission:", transmission_time, "waiting:", buffer_time)

        elif self.policy == "A2C_TEST":
            available_computing_resource = []
            waiting_task_list = []
            delay_list = []
            edge_manager = Simulator.get_instance().get_scenario_factory().get_edge_manager()
            edge_list = edge_manager.get_node_list()
            link_list = edge_manager.get_link_list()
            topology = edge_manager.get_network()

            for edge in range(len(edge_list)):
                route = topology.get_path_by_dijkstra(source, edge + 1)
                delay = 0

                for idx in range(len(route)):
                    if idx + 1 >= len(route):
                        break

                    for link in link_list:
                        link_status = link.get_link()
                        set = [route[idx], route[idx + 1]]

                        if sorted(set) == sorted(link_status):
                            delay += link.get_delay()
                            break

                delay_list.append(delay)

            for edge in edge_list:
                waiting_task_list.append(len(edge.get_waiting_list()) / 100)
                available_computing_resource.append(edge.CPU)

            max_resource = max(available_computing_resource)
            resource_list = []

            for i in range(len(edge_list)):
                resource_list.append(available_computing_resource[i] / max_resource)

            state_ = [task.get_remain_size() / 1000] + [task.get_task_deadline() / 1000] + \
                     resource_list + waiting_task_list + delay_list
            state = np.array(state_, ndmin=2)

            # edit
            action = self.agent.sample_action(state)
            self.action = np.array(action, ndmin=2)
            self.state = state
            action_sample = np.random.choice(Simulator.get_instance().get_num_of_edge(), p=np.squeeze(action))
            collaborationTarget = action_sample

        return collaborationTarget

    def save_weight(self):
        self.agent.policy.save_weights(self.file_path)

    def shutdown(self):
        if self.training_enable:
            self.loss_file.close()

    def training(self):
        if self.training_enable and Simulator.get_instance().get_warmup_time() < Simulator.get_instance().get_clock():
            print("**************" + str(self.epoch) + "***********")

            # for epc in range(self.epoch):
            if len(self.state_list) > 0:
                critic1_loss, critic2_loss, actor_loss, alpha_loss = self.agent.train(self.state_list,
                                                                                      self.actions_list,
                                                                                      self.rewards_list,
                                                                                      self.next_states_list)

                print("---------------------------")
                print("actor loss:", actor_loss.numpy(), "critic loss1:", critic1_loss.numpy(),
                      "critic loss2:", critic2_loss.numpy())
                self.loss_file.write("actor_loss: " + str(actor_loss.numpy()) +
                                     " critic_loss1: " + str(critic1_loss.numpy()) +
                                     " critic_loss2: " + str(critic2_loss.numpy()) + "\n")
                self.agent.update_weights()

                self.state_list = np.empty((0, self.state_space), dtype=np.float64)
                self.actions_list = np.empty((0, self.action_space), dtype=np.float64)
                self.rewards_list = np.empty((0, 1), dtype=np.float64)
                self.next_states_list = np.empty((0, self.state_space), dtype=np.float64)
