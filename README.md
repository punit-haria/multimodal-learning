#### Semi-Supervised Multimodal Learning with Generative Models

This repo contains accompanying code for my [master's thesis](http://punitshah.ca/docs/thesis.pdf). This includes implementations of a multimodal [variational autoencoder](https://arxiv.org/abs/1312.6114) (VAE), and incorporates variants of the [PixelCNN](https://arxiv.org/abs/1601.06759) architecture. The goal is to learn representations from multiple image modalities, and to provide a generative model for realizing plausible, new configurations in data space. See [jointvae.py](./code/models/jointvae.py) for a multimodal VAE implementation on **image** data, and see [multimodalvae.py](./code/models/multimodalvae.py) for a multimodal VAE implementation on **image and language** data. Various deep neural network architectures for the VAE are implemented in [layers.py](./code/models/layers.py). 

For a more detailed exposition of the multimodal learning problem, please see my [master's thesis](http://punitshah.ca/docs/thesis.pdf) as well as this post: [http://punitshah.ca/posts/multimodal/](http://punitshah.ca/posts/multimodal/)

For a walkthrough applying these techniques to an image dataset, please have a look here: [http://punitshah.ca/projects/multimodal_example/](http://punitshah.ca/projects/multimodal_example/) 
