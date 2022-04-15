from ecos.simulator import Simulator
from ecos.network_model import Network_model
from ecos.event import Event
from ecos.log import Log


class CloudManager:
    def __init__(self, cloud_props, network_props):
        self.node_list = list()
        self.cloud_props = cloud_props
        self.cloud_network_props = network_props
        self.cloud_network = None
        # 1 : FINISHED, 2 : RUNNABLE
        self.state = 1

        #minseon
        self.cloud_id = 0

    #minseon
    def get_cloud_id(self):
        return self.cloud_id

    def get_node_list(self):
        return self.node_list

    def get_state(self):
        return self.state

    def start_entity(self):
        if self.state == 1:
            self.state = 2

        self.create_cloud_server()

        return True

    def shutdown_entity(self):
        if self.state == 2:
            self.state = 1

        return True

    def create_cloud_server(self):
        #
        cloud = Cloud(0, self.cloud_props, Simulator.get_instance().get_clock())
        self.node_list.append(cloud)
        self.cloud_network = Network_model(-1, 0, int(self.cloud_network_props["bandwidth"]),
                                           int(self.cloud_network_props["propagation"]))
        print("Create cloud server")

    def receive_task(self, event):
        cloud = self.node_list[0]

        cloud.task_processing(event.get_task())

    def get_cloud_network(self):
        return self.cloud_network


class Cloud():
    def __init__(self, id, props, time):
        self.CPU = props["mips"]
        self.id = id
        self.exec_list = list()
        self.finish_list = list()
        self.waiting_list = list()
        self.previous_time = time

    def get_cloud_id(self):
        return self.id

    def task_processing(self, task):
        self.exec_list.append(task)

        msg = {
            "task": "check",
            "detail": {
                "node": "cloud",
                "id": 0
            }
        }


        expected_finish_time = self.CPU / len(self.exec_list)

        event = Event(msg, None, round(expected_finish_time, 3))
        self.previous_time = Simulator.get_instance().get_clock()
        Simulator.get_instance().send_event(event)

    def update_task_state(self, simulationTime):
        timeSpen = simulationTime - self.previous_time
        allcated_Resource = self.CPU / len(self.exec_list)

        for task in self.exec_list:
            remain_size = round(task.get_remain_size() - (allcated_Resource * timeSpen), 0)
            task.set_remain_size(remain_size)
            task.update_finish_time(timeSpen)

        if len(self.exec_list) == 0 and len(self.waiting_list) == 0:
            self.previous_time = simulationTime
            return

        for task in self.exec_list:
            if task.get_remain_size() <= 0:
                self.exec_list.remove(task)
                self.finish_list.append(task)
                self.finish_task(task)

        if len(self.exec_list) > 0:
            # add event
            nextEvent = 99999999999999
            for task in self.exec_list:
                remainingLength = task.get_remain_size()


                estimatedFinishTime = (remainingLength / allcated_Resource)

                if estimatedFinishTime < 0.001:
                    estimatedFinishTime = 0.001

                if estimatedFinishTime < nextEvent:
                    nextEvent = estimatedFinishTime

            msg = {
                "task": "check",
                "detail": {
                    "node": "cloud",
                    "id": 0
                }
            }
            event = Event(msg, None, nextEvent)
            Simulator.get_instance().send_event(event)

        self.previous_time = simulationTime

    def finish_task(self, task):
        task.set_finish_node(0)
        task.set_processing_time(self.previous_time, 1)
        task.set_end_time(self.previous_time)
        Log.get_instance().record_log(task)
        self.finish_list.remove(task)

    def get_exec_list(self):
        return self.exec_list

    def get_cloud_CPU(self):
        return self.CPU



