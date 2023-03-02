# DeepSearch

This is meant to be a combination of well-known search algorithms (e.g. minimax, A*, etc.) with neural networks as approximators for the evaluation function(s).


## Current ideas to explore

* MCTS as an alternative to min-max
* Improving the self-play and backpropagation strategy:
  * Maybe it would be better to use a different network to do the data generation (off-policy learning) and a different for back-propagation.
  * After the learning steps the two networks should compete and the winner gets to be/remain the network that generates actions (idea for policy networks).
  * Maybe we should first generate some data and the separately do supervised learning on them for some number of steps instead of continuously learning.

## References

1. https://www.davidsilver.uk/wp-content/uploads/2020/03/bootstrapping.pdf
2. http://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26646701.pdf
3. https://github.com/algerbrex/blunder/tree/main
4. https://matthewdeakos.me/2018/07/03/integrating-monte-carlo-tree-search-and-neural-networks/
5. https://www.mdpi.com/2076-3417/11/5/2056
