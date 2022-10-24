# Quasifully Supervised Learning

Evaluation of the quasifully supervised learning concept first authored here:

> [Jie Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+J), [Chengchao Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+C), [Yezhou Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Y), [Yang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Mingli Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+M),  
_Transductive Unbiased Embedding for Zero-Shot Learning_,  
[arXiv:1803.11320](https://arxiv.org/abs/1803.11320) [cs.CV].

Visual encoder used in models:

> [Mingxing Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+M), [Quoc V. Le](https://arxiv.org/search/cs?searchtype=author&query=Le%2C+Q+V),  
_EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks_,  
[arXiv:1905.11946](https://arxiv.org/abs/1905.11946) [cs.LG]

## Datasets

### [Animals with Attributes 2](https://cvml.ist.ac.at/AwA2/) (images in JPEG format of 50 animals classes)

Animals with Attributes2 Dataset, rc1, April 14th 2017 Christoph H. Lampert <chl@ist.ac.at> http://cvml.ist.ac.at/AwA2

Per `README-attributes.txt` and `README-images.txt`:

> Animal/attribute matrix for 50 animal categories and 85 attributes. Animals and attributes are in the same order as in the text files `classes.txt` and `predictes.txt`. The numeric data was originally collected by Osherson et al. [1], and extended by Kemp et al. [2]. Missing values in the numeric table are marked by $-1$. The binary matrix was created by thresholding the continuous table at the overall mean.

> __NOTE__: The images in this archive are NOT the ones used in the original publications [3,4]. 

> __DISCLAIMER__: The images in this archive are NOT the ones of the original "Animals with Attributes (AwA)" dataset http://cvml.ist.ac.at/AwA from the publications [3,4]. Those images are currently not publicly available.  

> The dataset is constructed as a plug-in replacement of the AwA dataset, and it has similar charactersitics as the original "Animals with  Attributes" dataset  (same classes, similar number of images per classes, ...). The images in this archive were collected in 2016 from public web sources (Flickr, Wikimedia,...). While collecting, we made sure that all images were accompanied by a license that allows free use and redistribution, see the included "licenses" folder. In the case that you hold the copyright to one or more of the included images and you did not actually release them under the indicated license, please let us know and we will update the information or remove the image from the collection. As far as possible, we made sure that there is no overlap between the new Animals with Attributes 2 (AwA2) dataset and the images of the original Animals with Attributes (AwA) dataset.  

>[1] D. N. Osherson, J. Stern, O. Wilkie, M. Stob, and E. E. Smith. _Default probability_. Cognitive Science, 15(2), 1991.  
[2] C. Kemp, J. B. Tenenbaum, T. L. Griffiths, T. Yamada, and N. Ueda. _Learning systems of concepts with an infinite relational model_. In AAAI, 2006.  
[3] C. H. Lampert, H. Nickisch, and S. Harmeling. _Learning To Detect Unseen Object Classes by Between-Class Attribute Transfer_. In CVPR, 2009  
[4] C. H. Lampert, H. Nickisch, and S. Harmeling. _Attribute-Based Classification for Zero-Shot Visual Object Categorization_, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), p 453-465, 2014  

__ATTENTION__: Data images have to be manually downloaded
-   Download image archive [here](https://cvml.ist.ac.at/AwA2/AwA2-data.zip),
-   After extracting, images should be in `./datasets/animals_with_attributes/JPEGImages/*/*.jpg`.
