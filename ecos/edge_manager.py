from ecos.edge import Edge
from ecos.event import Event
from ecos.simulator import Simulator
from ecos.orchestrator import Orchestrator
from ecos.topology import Topology
from ecos.network_model import Network_model
import math

# 22.01.05
class EdgeManager:
    def __init__(self, edge_props, edge_network_props):
        self.node_list = list()
        self.edge_props = edge_props
        self.edge_network_props = edge_network_props
        self.edge_network = None
        self.edge_link_list = list()
        # 1 : FINISHED, 2 : RUNNABLE
        self.state = 1
        self.orchestrator = None
        self.waiting_task = list()
        self.epoch = 1
        self.orh = Orchestrator(Simulator.get_instance().get_orchestration_policy(), 1)

    #minseon
    def get_node_list(self):
        return self.node_list

    def get_state(self):
        return self.state

    def start_entity(self):
        if self.state == 1:
            self.state = 2

        self.create_edge_server()

        msg = {
            "task": "training",
            "detail": {
                "node": "edge"
            }
        }

        event = Event(msg, None, 2)
        Simulator.get_instance().send_event(event)

        return True

    def shutdown_entity(self):
        if self.state == 2:
            self.state = 1

        self.orh.shutdown()


        return True

    def create_edge_server(self):
        id = 1

        for i in range(len(self.edge_props)):
            #och = Orchestrator(Simulator.get_instance().get_orchestration_policy(), id)
            edge = Edge(id, self.edge_props[i], 0, 3)
            id += 1
            self.node_list.append(edge)

        self.edge_network = Topology()
        self.edge_network.link_configure(self.edge_network_props)

        # create link
        for config in self.edge_network_props["topology"]:
            networkModel = Network_model(int(config["source"]), int(config["dest"]),
                                         int(config["bandwidth"]),
                                         int(config["propagation"]))

            self.edge_link_list.append(networkModel)

    def receive_task_from_edge(self, event):
        # find edge
        msg = event.get_message()
        task_list = list()

        for edge in self.node_list:
            task_num = len(edge.get_exec_list()) + len(edge.get_waiting_list())
            task_list.append(task_num)

        data = [x ** 2 for x in task_list]
        if sum(data) != 0:
            load_balance = (sum(task_list)**2) / sum(data)
        else:
            load_balance = 1
        task = event.get_task()
        task.set_load_balance(load_balance)

        node = self.node_list[int(msg["detail"]["route"][0]) - 1]
        node.task_processing(task)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def receive_task_from_device(self, event):
        #
        task = event.get_task()
        self.offloading(task)

    def offloading(self, task):
        source_edge = task.get_source_node()
        node = self.node_list[source_edge - 1]
        dest = self.orh.offloading_target(task, source_edge)
        task.set_processing_node(dest)
        task.set_status(1)
        threshold = 0
        x = 0

        #TaskClassification



        for i in range(len(node.get_waiting_list())):
            sum = 0
            if i >= 2:
                buffer_time = task.get_input_size() / node.get_CPU()
                for j in range(len(node.get_waiting_list())-1):
                    sum = buffer_time + sum

            x = task.get_task_deadline() - (task.get_input_size() / node.get_CPU()) - sum

        threshold = self.sigmoid(x) * node.get_CPU()

        if threshold > task.get_input_size()/task.get_task_deadline():
            dest = source_edge

        else:
            dest = self.orh.offloading_target(task, source_edge)



        # calculate network delay
        # network module does not complete
        if dest == source_edge:
            msg = {
                "network" : "transmission",
                "detail": {
                    "source" : -1,
                    "route" : [source_edge]
                }
            }

            event = Event(msg, task, 0)
            self.receive_task_from_edge(event)
        elif dest == 0:
            # collaboration target is cloud
            cloudManager = Simulator.get_instance().get_scenario_factory().get_cloud_manager()
            network = cloudManager.get_cloud_network()
            delay = network.get_download_delay(task)

            msg = {
                "network": "transmission",
                "detail": {
                    "source": source_edge,
                    "type": 0,
                    "link": network,
                    "delay": delay
                }
            }

            evt = Event(msg, task, delay)

            Simulator.get_instance().send_event(evt)
        else:
            route_list = self.edge_network.get_path_by_dijkstra(source_edge, dest)
            dest = route_list[1]
            set = [source_edge, dest]
            delay = 0

            # find link
            for link in self.edge_link_list:
                link_status = link.get_link()

                if sorted(set) == sorted(link_status):
                    delay = link.get_download_delay(task)

                    msg = {
                        "network": "transmission",
                        "detail": {
                            "source": source_edge,
                            "type": 1,
                            "link": link,
                            "route": route_list,
                            "delay": delay,
                        }
                    }

                    evt = Event(msg, task, delay)

                    Simulator.get_instance().send_event(evt)
                    break

    def get_network(self):
        return self.edge_network

    def get_link_list(self):
        return self.edge_link_list

    def get_training(self):
        self.orh.training()

#이거안되면 SSD 사자 민선