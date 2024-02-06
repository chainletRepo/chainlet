# Bitcoin Transaction Graph Analysis

This folder involves downloading and analyzing the complete Bitcoin transaction graph from its inception in 2009 until 2023 using the Bitcoin Core Wallet and the Bitcoin-ETL library. The objective is to extract daily transactions on the network, construct the Bitcoin graph and provide publicly available address orbits data.

## Data Source

The data is sourced from the Bitcoin Core Wallet and the Bitcoin-ETL library. The address orbits, which are 27 GB in size, are publicly available at [***.org/btc/orbits](***.org/btc/orbits).

We also include two additional label datasets for further analysis:

- Ransomware Addresses: Our ransomware dataset is a union of datasets from three widely adopted studies: Montreal [1], Princeton [2] and Padua [3]. The combined dataset contains addresses from 27 ransomware families. 19930 unique ransomware addresses appear a total of 62641 times between 2009-2023.

- Darknet Addresses: We have downloaded the Grams dataset from Darknet Market Archives [4]. Grams was a platform that primarily focused on facilitating searches for market listings. The platform utilized a variety of methods to acquire the listings, including utilizing API exports provided by various markets. The dataset spans from June 9, 2014, to July 12, 2015. 7557 unique addresses associated with darknet marketplaces appear a total of 1288100 times.

[1] Masarah Paquet-Clouston, Bernhard Haslhofer, and Benoit Dupont. 2018. Ransomware payments in the bitcoin ecosystem. arXiv preprint arXiv:1804.04080
(2018).

[2] D. Y. Huang and D. McCoy. 2018. Tracking Ransomware End-to-end. In Tracking Ransomware End-to-end. IEEE, 1–12.

[3] Mauro Conti, Ankit Gangwal, and Sushmita Ruj. 2018. On the economic significance of ransomware campaigns: A Bitcoin transactions perspective. Computers
& Security (2018).

[4] Gwern Branwen. 2023. Darknet Market Archives. Online, Vol. 1. https://gwern.net/DNM-archives#grams

## Methodology

The project utilizes a 24-hour time window centered on the Central Standard Time zone to extract daily transactions on the network and construct the Bitcoin graph. The address orbits are used without any normalization or preprocessing, as extracting orbits is not demanding in terms of computational resources.

## Results

The results of the analysis are publicly available at [***.org/btc/orbits](***.org/btc/orbits) for anyone interested in further research or analysis.

## Acknowledgements

We would like to thank the developers of the Bitcoin Core Wallet and the Bitcoin-ETL library for making this project possible.

## Contact

For questions or comments about this project, please contact us at [***.org/btc/contact](***.org/btc/contact).
