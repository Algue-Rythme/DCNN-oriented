import tensorflow as tf

import labelized_graph


if __name__ == "__main__":
    tf.enable_v2_behavior()
    graph = labelized_graph.DenseLabelizedGraph()
    graph.load_from_text("toy_graph.txt")
    graph.summary()
