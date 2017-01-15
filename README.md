### Why this? What is the goal?

The goal of this repository is to write all the recurrent architectures from scratch in tensorflow for learning purposes. This is a Work-In-Progress. I plan to implement some more architectures and publish the results and performances for all of them. The inspiration for this post was the last paragraph of this post: [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) Chris Olah mentioned two papers that did extensive study on recurrent architectures and I wanted to implement all the architectures in these two papers. A short Google search resulted that Jim Flemming already did half the work [here](https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93#.89xd4s9ii), so I decided to implement all the remaining architectures of Jozefowicz's paper. (I also updated parts of his code so that all the architectures work in the newest version of tensorflow. Both these papers are fantastic and worth a read. Feel free to send me a pull request if you spot an error and/or find other papers with recurrent architecture variants. As and when time permits, I will implement them. All the implementations are in Tensorflow (0.12).

### Deep Learning Recurrent Architectures
 -  [LSTM Network Variants](https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93#.89xd4s9ii) This tutorial has a very nice approach to creating variations of LSTM Networks. A good approach to learning how to code a new network architecture and more importantly a methodical approach to understanding the gates in LSTM
 -  [Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)

This was mainly because I wanted to learn the actual implementations of various recurrent neural network architecures and implement them from scratch without using pre defined lstm, gru etc. This is directly a fork of [LSTM Network Variants](https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93#.89xd4s9ii), with the code changes to run on the most recent version of tensorflow. (0.12.0 as of this writing). I will keep this repositiory upto date with the new changes.

Also this repo has more network architectures from here: [Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)

The implementations are not optimal, in the sense, that in the actual implementations of the LSTM, GRU and RNN cells the states and input are concatenated before multiplications to reduce the number of matrix multiplications whereas this is directly an implementation of the lstm network that you would see in a textbook.


### Other Tutorials with that are also helpful
 - [RNN Network implementation Tensorflow](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767#.qk7xgwwa4) A series of post on implementing recurrent neural network architectures. A good follow up post would be the next article.
 - [Reinforcement Learning](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.tpmlehy2p) A fantastic hands-on tutorial on reinforcement learning using the openAI platform.




## Recurrent Architectures Implemented

If with a (*) then it was implemented in [LSTM Network Variants](https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93#.89xd4s9ii), else was implemented by me based on [Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) . Also network architectures that I have implemented follow the conventions and syntax of [Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf). 

- __mut1__ : Variant 1 from [Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- __mut2__ : Variant 2 from [Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- __mut3__ : Variant 3 from [Empirical Exploration of Recurrent Network Architectures](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- __vanillaRNN__ : Just a vanilla RNN Network
- __gru__ : Gated Recurrent Unit
- __cifg__ (*) : Coupled input-forget gate
- __fgr__ (*) : Full Gate Recurrence
- __lstm__ (*) : Long Short Term Memory
- __nfg__ (*) : No forget gate
- __niaf__ (*) : No input activation function
- __nig__ (*) : No input gate
- __noaf__ (*) : No output activation function
- __nog__ (*): No output gate
- __np__ (*): No peephole connections


### Instructions

See the jupyter notebook here: https://github.com/debajyotidatta/RecurrentArchitectures/blob/master/Empirical%20Exploration%20of%20Recurrent%20Network%20Architectures.ipynb
