# Netflixer 

![General screenshot of our application.](GeneralScreenshot.png)

## Team members
* Zhuorui Li : Master of Science in Sustainable Design, zhuoruil@andrew.cmu.edu
* Pengcheng Li (Jacob) : Master of Entertainment Technology, pengchel@andrew.cmu.edu
* Hyungil Kim (Ed) : Master of Information System Management, hyungilk@andrew.cmu.edu 

## Abstract

Trying to create contents to show on Netflix? You are in luck. Check out our website to learn how Netflix distributs its tv shows and movies across the world. Find out possible insights to help you decide what kind of show to produce or even which director to partner. Depending on what content question you are curious about Netflix, we try to show it to you.

## Goal of our project

Our goal is to help users understand how Netflix distributes its shows across the world from the start of time to now. Especially, this will help people who want to establish their content creation strategy in near future according to key aspect(e.g. country, genre, target audience.)

## Overview of development process

We started looking for possible datasets to work on. We looked through all the necessary datasets available and other datasets used for competitions on [Kaggle](https://www.kaggle.com/shivamb/netflix-shows). Eventually, we decided to choose this Netflix shows dataset as we all use Netflix and we are curious how frequently Netflix releases its tv shows and movies around the world and possibly find and predict trends of Netflix’s projections in the future. On top of that, given that this dataset has around 8000 rows of data, we believe it is just the right amount of data needed for us to explore and generate something useful.

Our team consist of three member and each member takes charge of each part, part1-Zhouri, part2-Jacob, and part3-Ed.
Zhouri has focused on exploratory data analysis and implementation of interactive filtering with combination of two graphs.
Jacob takes data deep dive part to extract more information on combination or directors, genres, and its trend on country and year according to selected category.
Ed tries to visualize the similarity of each country in term of media age rating and enables users to compare two countries which have similar pattern and to research further more on these countries.

In total, we nearly used 20 hours for this project from selecting an interesting dataset on our own to finalize the report.  

## Rationals for our design

We split our design for this project into 4 main parts.

For part 0, it is designed for users who may want to understand how we processed the data to reach all the diagrams shown.

For part 1, we generated all the general visualizations for users to learn more about the general trend of shows in Netflix across the years, countries, TV ratings and genre. This allows the user to understand what Netflix focuses on in terms of production.

For part 2, we have decided to show more in-depth analysis of the dataset for users to select the top N number of a category for further analysis. For example, if the users are curious what are the top 10 countries or genres that have been showing in Netflix, they can simply turn up the slider accordingly to find this data.

For part 3, we allow users to further deep dive into the data by comparing between 2 variables. In this case, we have chosen countries. If they are interested in finding out how many shows are produced between certain countries, they can simply choose them and a comparison chart will be shown either against the world or against the other country.


## Success story of our project

In this project, we provide a multiple view of Netflix's media contents for contents creating studios and broad casting firm and enable them to establish proper media creation strategy in term of genre, country, and its trend. 


#### Q1.How Nexflix data looks likes and how each feature relates year of release?
At first, we can see the overall content distribution in Netflix with the entire date. And also we can use the interative filter for each graph on the left and the graph on the right will change accordingly.  

![img_ratio](https://user-images.githubusercontent.com/79838132/156280731-bc4777f9-7f08-48ee-b941-0b382d0efde3.PNG)  

Then, users can see the `analyzing releasing year and month` for different features as below.
![img_eda_1](https://user-images.githubusercontent.com/79838132/156281088-bcd0bb78-c865-4f0e-8c1a-1c3a1f850840.PNG)  
![img_eda_2](https://user-images.githubusercontent.com/79838132/156281126-c4e97fd4-6eee-44a8-ae90-8e5d4e9bb6ea.PNG)  
![img_eda_3](https://user-images.githubusercontent.com/79838132/156281149-1305e36b-9e70-495c-b719-e548e6f56e6f.PNG)

During this process, the content creator or studio can discover how each media is associated with different features, such as added month, contents rating, age group(aggregated version of rating), and country. For example, the user will clearly find out how movie is more populat on the Netflix platform over the years and TV series is only added at a later stage. Also, the main targeted age group for Netflix shows are Adults and Teens with a strong reliance in countries like United States, India, United Kindow.

#### Q2. According to a category, which genre and media target audience would be changed and check their records over year
Secondly, in part2, users can choose a category to deep dive, for example they can choose director category and can see the top N directors in media, their genres, media age group, popular contries, and release year of their movies. This part helps us to discover more detailed view for each category. For example, director view provides their preference of genres, target audience, popular country, and their contents release year.

![img_part2_1](https://user-images.githubusercontent.com/79838132/156282437-3b1d578c-6762-4aa9-9438-31b91f0f8cab.PNG)  
![img_part2_2](https://user-images.githubusercontent.com/79838132/156282493-2bdebca8-b8a0-46fa-9b81-7f850d38ab1e.PNG)

For media content creators, they may want to learn how certain directors can be so successful on Netflix. Hence, they can try to see what kind of show they are creating and follow suit. In part 2, the top directors are focusing greatly on the Adult genre and Comedies. In terms of country distribution, althought United States still remain as the highest number of films produced for these directors, the second most is actually Phillippines. The media content creators may then consider finding directors catering to their needs in creating shows in those countries.

#### Q3. Which country has similar with the others and can we specify some countries to discover more?
In part3, we try to compare each country's media characteristic, especially media type. As you can see below image, some countries show similar contents composition, such as Spain and Mexico, the US and UK, and Japan and South Korea. As you can see this heatmap, we might want to compare two countries from the above examples. It will help people who want to see media consumption trend in those country and can establish similar curating strategy for these countries.  
In this case, the user can try to find the similarity between Spain and Mexico with already specified genre and director from part 2. This process can help users to narrow down for their future research scope.

![img_part3_1](https://user-images.githubusercontent.com/79838132/156283674-8c96087c-2a45-45a9-b9b2-e3e75b64e917.PNG)

In the end, user can specify a country to compare its media trend compared to global level, so it enables to check whether the country is following global trend or not.  

![img_part3_2](https://user-images.githubusercontent.com/79838132/156283714-79ffff20-5576-4b71-8f15-99da4acb68f0.PNG)

In summary, this project enables people who work in media content studio or creators to check the trend of media availability at Netflix and the relation between features and trend. Also, they can discover more deeply on director, country, and genre and get the detail information on each category. Finally, we try to this project with their future work by specifying some countries which has simliar pattern in target audience.
