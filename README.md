# NIH RePORTER

## Summary

I started this project with the goal of learning some NLP and applying it to looking at NIH Funding. The project trains gensim's implementation of word2vec on NIH grant abstracts and creates TF-IDF document vectors. 

With this dataset in hand, one can    
- test word2vec representation classics like king - man + woman = queen with multiple_sclerosis - human + mouse = ?. Or A grant document vector - method1 + method2 -> ?. 
- Find TF-IDF keywords for grants by looking at the nearest words to the grant's document vector
- Find nearest neighbor grants by looking at the nearest document vectors to a grant's document vector
- cluster grants into unsupervised list of topic categories
- create nice tSNE visualizations of NIH grant funding

**Sample output**   

![20,000 NIH Grants visualized by tSNE](NIH_Grants_20000.png)

`word_vectors.most_similar(positive=['multiple_sclerosis','skin'],negative=['brain'])`  

```
    vulgaris
    vitiligo
    cutaneous
    atopic_dermatitis
    blister
```


`word_vectors.most_similar_cosmul(positive=['multiple_sclerosis','mouse'],negative=['human'])` 

```
    experimental_autoimmune
    cuprizone
    eae
    encephalomyelitis_eae
    mog
```

`query_grant(50000,grant_vectors_dedup)`

```
Project Title: 

Develop novel therapeutic strategy for brain tumor


Abstract: 

Mutations in isocitrate dehydrogenase (IDH1/2) are common genetic abnormalities in grade II and III diffusive astrocytomas and oligodendrogliomas. In WHO grade II/III gliomas, IDH mutated tumors are highly prevalent comprising nearly 80% of all clinical cases. In glioma, IDH mutations cluster in an arginine residue at the center of the catalytic domain (IDH1 R132, IDH2 R172). Mutant IDH confers neomorphic enzymatic activity that, catalyzes alpha-ketoglutarate (alpha-KG) into 2-hydroxyglutarate (2-HG), an oncometabolite closely related to the deactivation of alpha-KG-dependent deoxygenases. For example, IDH1 mutant derived 2-HG promotes hypoxia signaling by perturbing the catalytic activity of prolyl hydroxylase, resulting in constitutive activation of hypoxia-inducible factor 1alpha (HIF-1alpha). Additionally, 2-HG has also been found to affect collagen maturation and basement membrane function, which may facilitate cancer cell infiltration and promote glioma progression. Clinically, the occurrence of IDH mutations predicts longer survival and greater sensitivity to chemotherapy in low-grade gliomas and secondary glioblastomas. A phase III clinical trial has provided the direct link between IDH mutation and survival benefit from chemotherapy. Combined with O-6-methylguanine-DNA methyltransferase (MGMT) promoter methylation status, IDH mutations serve as an important prognostic marker for gliomas treated with radiation and chemotherapy. Although there has been increasing awareness of the correlation between IDH mutations and chemo-sensitivity, the molecular mechanism that determines the vulnerability that results from IDH mutations remains unanswered. DNA repair is defined as a series of molecular changes that occur in response to compromised chromosomal integrity. DNA repair is integral to cancer therapies based on generating DNA damage in chromosomal DNA, such as radiation therapy and cytotoxic chemotherapies. The activation and expression level of DNA repair pathways largely determines the efficacy of chemotherapies and resulting clinical outcomes. Several studies shed light on the changes in DNA repair mechanism in IDH1-mutated cells, including RAD51 and ATM pathways. The distinct connection between DNA repair pathway and chemo-sensitivity in IDH-mutated cells, however, must be further elucidated. In the present study, we established cell lines that stably expresses either the IDH1 R132C or IDH1 R132H pathogenic mutations. We demonstrate that our cell lines recapitulate the IDH1-mutant phenotypes found in glioma patients. The IDH1 mutations resulted in the metabolic reprogramming and cytotoxic effects via 2-HG production. In addition, cells with mutant IDH failed to form the poly (ADP-ribose) polymer (pADPR), and therefore were unable to maintain genomic integrity. Furthermore, targeting the PARP DNA repair mechanism remarkably potentiated the cytotoxic effects of chemotherapy. Taken together, our findings indicate a potential molecular mechanism of chemo-sensitization in IDH mutant gliomas and, suggests a novel therapeutic strategy for clinical therapies.


Most similar word vectors:
idh_mutation, 0.861701488494873
glioma, 0.7649029493331909
idh_mutant, 0.7618250250816345
idh, 0.753430962562561
mutant_idh, 0.7279563546180725
idh_mutate, 0.7206025719642639
isocitrate_dehydrogenase, 0.7024551630020142
oncometabolite, 0.6593918204307556
dehydrogenase_idh, 0.6246827244758606
mutation_isocitrate, 0.6220428347587585


Most similar grants: 
143175 Exploiting the vulnerabilities in mutant IDH gliomas, 0.9382784366607666
148052 Project 2 - Targeting IDH-mutant gliomas (Cahill/Kaelin), 0.9043545126914978
13761 Targeting Nicotinamide Adenine Dinucleotide (NAD+) metabolism in IDH mutant gliomas, 0.8862265348434448
102345 Novel Approaches to Modeling and Treating IDH1 Mutant Glioma, 0.8796610236167908
62149 Understanding the role of IDH in malignant gliomas, 0.8785685300827026
74333 The Role of IDH1 Mutations in Gliomagenesis and Metabolism, 0.8738013505935669
78787 Modeling low-grade gliomas using human pluripotent stem cells, 0.8685322999954224
49272 Exploiting Mutant IDH1/2-induced Homologous Recombination Defects in Cancer, 0.8644766807556152
43422 Elucidating and targeting the molecular foundations of IDH mutant glioma, 0.8513734936714172
114898 Elucidating and targeting the molecular foundations of IDH Mutant glioma, 0.8513734936714172
```



## To do
Here are some ideas that I still want to implement: 
- Level of funding to each cluster  
- Representation of different institutions or locations within each cluster  
- Identify faculty that are most diverse across clusters vs most focused  
- Rates of patent activity per cluster  
- Identify grants that are "unusual" combination of topic vectors  
- Map startups press release vectors onto grant space.  

## Setup

### Data

The NIH provides lots of grant data to the public at the [NIH Exporter Website](https://exporter.nih.gov/ExPORTER_Catalog.aspx). 

For the years 2018-2020, I downloaded the projects and abstracts to the `data/raw` directory. There should be 6 files of the format:  
- projects: `RePORTER_PRJ_C_FY[YEAR]_new.csv`
- abstracts: `RePORTER_PRJABS_C_FY[YEAR]_new.csv`

### Environment

This project required the following libraries, which can be found in the environment.yml file.  

#### Utils
- jupyter=1.0.0=py38h50d1736_6
- pandas=1.2.4=py38h1f261ad_0

#### Plotting 
- bokeh=2.3.1=py38h50d1736_0
- selenium=3.141.0=py38h5406a74_1002 

#### NLP 
- spacy=3.0.6=py38he35c9cc_0
- gensim=4.0.1=py38ha048514_0
- pyemd=0.5.1=py38h6be0db6_1002

I found spacy a pain to install with `conda`. This is what worked for me.    
``conda config --set channel_priority false   
conda update --all --yes   
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
``  

#### ML and tSNE
- scikit-learn=0.24.2=py38h011f2c5_0

## Notebooks

`1. Build and Clean Dataset`

This notebook cleans up and consolidates the NIH grant datasets. 

**Inputs:**
 - ../data/raw/ should contain csv datasets downloaded from NIH project exporter   
**Outputs:** 
 - ../data/clean/NIH_grants.csv should be cleaned and consolidated dataset
 
 
**Steps:**
- For each abstract, convert all words to lowercase and remove alphanumeric characters
- Lemmatize all words using en_core_web_sm model from gensim
- Create bigram model and learn word pairs
- Output single dataframe with all grant information and cleaned, lemmatized, bigrammed grant abstracts
 
Bigram models will be saved to ../models/. 

`2. Build Word2Vec Model`

Trains the word2vec model and creates document vectors for each grant. 

**Inputs:**
- data/clean/NIH_grants.csv output by first notebook   

**Outputs:** 
- data/clean/corpus.pkl, the corpus of abstracts in list of list form for training
- models/w2v.pkl, the trained word2vec model
- models/word_vectors.pkl, the word vectors, much faster than importing the whole model
- models/grant_vectors.pkl, document vectors for each grant

**Steps:**
- Construct corpus from the NIH grant abstracts in list of list form
- Train word2vec model
- Compute grant document vectors

We compute the grant vectors by doing a weighted average of the word vectors in each grant by the Term Frequency Inverse Document Frequency. 

First, Compute the TFIDF in order to create weights for averaging to create grant vectors. 

More information on TFIDF here: https://rare-technologies.com/pivoted-document-length-normalisation/

`3. Model Exploration and Clusters`

We have the following datasets and models now. 
1. `nih_grants` is pandas DataFrame that contains all the NIH grant data. Notable columns are `PROJECT_TITLE`, `PROJECT_TERMS`, `ORG_NAME`, `PI_NAMEs`, `TOTAL_COST`, `ABSTRACT_TEXT`

2. `word_vectors` is a gensim.models KeyedVectors instance for words identified in the cleaned and lemmatized abstracts. The keys are words. The vectors are (300,) vectors output by word2vec. 

3. `grant_vectors` is a gensim.models KeyedVectors instance with vectors assigned to each grant. This is calculated by averaging the vectors for each word in a the corresponding abstract. Keys are the index column in `nih_grants` DataFrame.

This notebook  
- Does some of the basic word2vec explorations described above
- Uses spectral clustering to cluster grant vectors of 20,000 random grants
- Compute tSNE of 20,000 random grants

`4. Analysis of Clusters`

Basic exploration of the clusters. 


