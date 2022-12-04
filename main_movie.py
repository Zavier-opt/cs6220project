from moviefederated_dm import MovieFederatedDM
from models.dnn import DNN
from p2pfl.node import Node
import time


def recommand_execution(n, start, simulation, conntect_to=None, iid=True):

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(
            DNN(),
            MovieFederatedDM(experiment="age",num_of_split=2, sub_id=i, number_sub=n,isRandom=True),
            simulation=simulation,
        )
        node.start()
        nodes.append(node)

    # Connect other network
    if conntect_to is not None:
        nodes[0].connect_to(conntect_to[0], conntect_to[1])

    # Node Connection
    nodes[1].connect_to(nodes[0].host,nodes[0].port)
    nodes[2].connect_to(nodes[1].host,nodes[1].port)
    nodes[3].connect_to(nodes[1].host,nodes[1].port)
    nodes[4].connect_to(nodes[1].host,nodes[1].port)
    nodes[5].connect_to(nodes[4].host,nodes[4].port)
    nodes[6].connect_to(nodes[4].host,nodes[4].port)
    nodes[7].connect_to(nodes[1].host,nodes[1].port)
    nodes[8].connect_to(nodes[7].host,nodes[7].port)
    nodes[9].connect_to(nodes[7].host,nodes[7].port)

    nodes[10].connect_to(nodes[0].host,nodes[0].port)
    nodes[11].connect_to(nodes[10].host,nodes[10].port)
    nodes[12].connect_to(nodes[10].host,nodes[10].port)
    nodes[13].connect_to(nodes[10].host,nodes[10].port)
    nodes[14].connect_to(nodes[10].host,nodes[10].port)
    nodes[15].connect_to(nodes[14].host,nodes[14].port)
    nodes[16].connect_to(nodes[14].host,nodes[14].port)
    nodes[17].connect_to(nodes[10].host,nodes[10].port)
    nodes[18].connect_to(nodes[17].host,nodes[17].port)
    nodes[19].connect_to(nodes[17].host,nodes[17].port)

    time.sleep(5)
    # for i in range(len(nodes) - 1):
    #     nodes[i + 1].connect_to(nodes[i].host, nodes[i].port, full=True)
    #     time.sleep(1)

    time.sleep(5)
    print("Starting...")

    for n in nodes:
        print(len(n.get_neighbors()))
        print(len(n.get_network_nodes()))

    # Start Learning
    if start:
        nodes[0].set_start_learning(rounds=1, epochs=200)
    else:
        time.sleep(20)

    # Wait 4 results
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break

    for node in nodes:
        node.stop()


if __name__ == "__main__":
    for _ in range(1):
        recommand_execution(20, True, True)
        #break