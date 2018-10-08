from pydot import Dot, Edge, Node
graph = Dot(graph_type='digraph')
graph.add_edge(Edge('test','test2'))
graph.add_edge(Edge('test','test3'))
graph.add_edge(Edge('test2','test4'))
graph.add_edge(Edge('test2','test5'))
graph.add_edge(Edge('test3','test6'))
graph.add_edge(Edge('test3','test7'))
graph.write_png('test.png')