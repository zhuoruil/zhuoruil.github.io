from itertools import count
from re import U
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load the data from CSV to a pandas dataframe and return it.
@st.cache
def load_data(path):
    return pd.read_csv(path)

@st.cache
def load_specific_data(df, show_type):
	return df[df['type'] == show_type]

# checks how much data is missing
def show_missing_data_statistics(df):
	for i in df.columns:
	    null_rate = df[i].isna().sum() / len(df) * 100 
	    if null_rate > 0 :
	        st.write(f"{i} null rate: {round(null_rate, 2)}%")


# fix missing data and returns a new df
def fix_missing_data(df):
	temp_df = df.copy()
	temp_df['country'].replace(np.nan, 'No Data', inplace=True)
	temp_df['cast'].replace(np.nan, 'No Data', inplace=True)
	temp_df['director'].replace(np.nan, 'No Data', inplace=True)

	# Drops
	temp_df.dropna(inplace=True)

	# Drop Duplicates
	temp_df.drop_duplicates(inplace=True)
	return temp_df


# currently too slow so deleted it. Consider again.
# @st.cache(suppress_st_warning=True)
def show_ratio_graph(df_to_use):
	fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.5))
	# use hardcord value since cache is not working
	# x = df_to_use.groupby(['type'])['type'].count()
	# y = len(df_to_use)
	# r = ((x / y)).round(2)
	# st.write(r)
	r = {'type': {'Movie': 0.70, 'TV Show': 0.30}}
	


	mf_ratio = pd.DataFrame(r).T
	ax.barh(mf_ratio.index, mf_ratio['Movie'], color='#b20710', alpha=0.9, label='Male')
	ax.barh(mf_ratio.index, mf_ratio['TV Show'], left=mf_ratio['Movie'], color='#221f1f', alpha=0.9, label='Female')

	ax.set_xlim(0, 1)
	ax.set_xticks([])
	ax.set_yticks([])

	for i in mf_ratio.index:
	    ax.annotate(f"{int(mf_ratio['Movie'][i]*100)}%", 
	                   xy=(mf_ratio['Movie'][i]/2, i),
	                   va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
	                   color='white')

	    ax.annotate("Movie", 
	                   xy=(mf_ratio['Movie'][i]/2, -0.25),
	                   va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
	                   color='white')

	for i in mf_ratio.index:
	    ax.annotate(f"{int(mf_ratio['TV Show'][i]*100)}%", 
	                   xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, i),
	                   va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
	                   color='white')
	    ax.annotate("TV Show", 
	                   xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, -0.25),
	                   va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
	                   color='white')

	# remove border in plot
	for s in ['top', 'left', 'right', 'bottom']:
		ax.spines[s].set_visible(False)
		ax.legend().set_visible(False)
	st.pyplot(fig, ax)


st.title("Netflix Titles Dataset Explorable and Visualization")

# defines the file path
file_path = "netflix_titles.csv"
with st.spinner(text="Loading data..."):
    df = load_data(file_path)

if st.checkbox("Feel free to check the raw data frame if you are ineterested"):
    st.write(df)

########## Part 0: data preprocessing and enhancements
st.header("Part 0: Data preprocessing and enhancements")

### Data fixing to prevent nulls
# find missing data statistics
if st.checkbox("A list of missing data statistics across columns"):
	show_missing_data_statistics(df)
	st.write(df.isna().sum())

# fix missing data with No Data
st.subheader("Fix missing data statistics across columns")
filtered_df = fix_missing_data(df) # use filtered_df from this point onwards

# show_missing_data_statistics(filtered_df) # empty means all fixed
st.write("Replace missing nan data with No Data text")
if st.checkbox("Check updated missing data rate"):
	st.write(filtered_df.isna().sum())

st.subheader("Preprocess data to add more columns")

# split date_added to month and year for reference
filtered_df["date_added"] = pd.to_datetime(filtered_df['date_added'])
filtered_df["date_added_month"] = filtered_df["date_added"].dt.month.fillna(0).astype('int64')
filtered_df['month_added']=filtered_df['date_added'].dt.month
filtered_df['month_name_added']=filtered_df['date_added'].dt.month_name()
filtered_df['year_added'] = filtered_df['date_added'].dt.year

# use the first country of the list
filtered_df['first_country'] = filtered_df['country'].apply(lambda x: x.split(",")[0])

# convert group rating number to age group
Kids = ['TV-Y','G']
Older_Kids = ['TV-Y7','TV-Y7-FV','PG']
Teens = ['TV-G','TV-PG','TV-14','PG-13']
Adults = ['TV-MA','NC-17','R','UR','NR']
def agegroup(x):
  if x in Kids:
    return "Kids"
  elif x in Older_Kids:
    return "Older Kids"
  elif x in Teens:
    return "Teens"
  elif x in Adults:
    return "Adults"
  else:
    return "Others"

filtered_df['age_group'] = filtered_df['rating'].apply(lambda x: agegroup(x))

if st.checkbox("Check enhanced columns"):
	st.write("Added columns include: date, month, year, age group and first country data")
	st.write(filtered_df.columns)

########## Part 1: basic stats data visualization and interaction
# please change df to filtered_df

# disable alt 5k rows limit
alt.data_transformers.disable_max_rows()

st.header("Part 1: Basic visualization of the overall dataset")

st.subheader("Netflix color scheme")
netflix_colors = ['#b20710', '#564d4d', '#221f1f', '#F5F5F1']
sns.palplot(netflix_colors)
st.pyplot(plt)

st.subheader("Distribution between Movie and Tv show")
show_ratio_graph(filtered_df) # very slow even with cache...

# make released year and month area count chart
#year 
year_chart = alt.Chart(filtered_df).mark_area(
    tooltip=True
).encode(
    alt.X("release_year", scale=alt.Scale(zero=False)),
    alt.Y('count()', scale=alt.Scale(zero=False)),
    alt.Color("type", scale=alt.Scale(domain=["Movie", "TV Show"], range=netflix_colors))
).transform_filter(
	'datum.release_year > 1970'
).properties(
	width=500,
	height=300,
	title='Netflix year and count chart from 1970s onwards'
)

#month
date_added_month_chart = alt.Chart(filtered_df).mark_area(
    tooltip=True
).encode(
    alt.X("date_added_month", scale=alt.Scale(zero=False)),
    alt.Y('count()', scale=alt.Scale(zero=False)),
    alt.Color("type", scale=alt.Scale(domain=["Movie", "TV Show"], range=netflix_colors))
).properties(
	width=500,
	height=300,
	title="Netflix months and count chart from Jan to Dec"
)


# TODO: add selection to this chart
# organize the two chart
st.subheader('Analyzing releasing year and month')
# row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns((.1, 2, 4, 2, .1))
# col1, col2 = st.columns(2)
# with row3_1:
#     st.write(year_chart)

# with row3_2:
#     st.write(date_added_month_chart)

selection = alt.selection_interval()

year_chart.add_selection(selection).encode(
    color=alt.condition(selection, "type", alt.value("grey"))
) | date_added_month_chart.transform_filter(selection)

#interactive chart
scatter = alt.Chart(filtered_df).mark_point(
    tooltip=True
).encode(
    alt.X("release_year", scale=alt.Scale(zero=False)),
    alt.Y("date_added_month", scale=alt.Scale(zero=False)),
    alt.Color("type", scale=alt.Scale(range=netflix_colors))
).transform_filter(
	'datum.release_year > 1970'
).properties(
	title="Netflix year and month distribution",
    width=500,
    height=300
)
hist = alt.Chart(filtered_df).mark_bar(
    tooltip=True
).encode(
    alt.X("release_year"),
    alt.Y("count()", type="quantitative"),
    alt.Color("type", scale=alt.Scale(domain=["Movie", "TV Show"], range=netflix_colors))
).transform_filter(
	'datum.release_year > 1970'
).properties(
	title="Netflix year distribution",
    width=500,
    height=300
)
selection = alt.selection_interval()


row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns((.1, 2, 4, 2, .1))
# with row4_1:
#     st.subheader("Scatter chart")
# with row4_2:
#     st.subheader("Hist chart")
st.write(selection = alt.selection_interval())
scatter.add_selection(selection).encode(
    color=alt.condition(selection, "type", alt.value("grey"))
) | hist.transform_filter(selection)
#rating
ratingbar = alt.Chart(filtered_df).mark_bar(
    tooltip=True
).encode(
    alt.X("count()", scale=alt.Scale(zero=False)),
    alt.Y("rating", scale=alt.Scale(zero=False)),
    alt.Color("type", scale=alt.Scale(domain=["Movie", "TV Show"], range=netflix_colors))
).properties(
	title="Netflix Ratings Distribution",
    width=500,
    height=300
)

# age group selction
st.write(selection = alt.selection_interval())
ratingbar.add_selection(selection).encode(
    color=alt.condition(selection, "type", alt.value("grey"))
) | hist.transform_filter(selection)
age_group = alt.Chart(filtered_df).mark_bar(
    tooltip=True
).encode(
    alt.X("age_group", scale=alt.Scale(zero=False)),
    alt.Y('count()', scale=alt.Scale(zero=False)),
    alt.Color("type", scale=alt.Scale(domain=["Movie", "TV Show"], range=netflix_colors))
).properties(
	title="Netflix Age Group Distribution",
    width=500,
    height=300
)

st.write(selection = alt.selection_interval())
age_group.add_selection(selection).encode(
    color=alt.condition(selection, "type", alt.value("grey"))
) | hist.transform_filter(selection)
#country part
countries_expand_df = df["country"].str.split(",", expand=True)
countries_df = df.copy()
countries_df = pd.concat([countries_df, countries_expand_df], axis=1)
countries_df = countries_df.melt(id_vars = ["type","title"], value_vars = np.arange(12), value_name="country")
countries_df = countries_df[countries_df["country"].notna()]

top20df = countries_df["country"].value_counts()[:20]
topcountry = countries_df[countries_df.country.isin(top20df.index)]
country = alt.Chart(topcountry).mark_bar(
    tooltip=True
).encode(
    alt.X("country",sort = "-y"),
    alt.Y("count()"),
    alt.Color("type", scale=alt.Scale(domain=["Movie", "TV Show"], range=netflix_colors))
).properties(
	title='Netflix top 20 countries distribution'
)

st.write(selection = alt.selection_interval())
country.add_selection(selection).encode(
    color=alt.condition(selection, "type", alt.value("grey"))
) | hist.transform_filter(selection)

# country = alt.Chart(countries_df).mark_bar(
#     tooltip=True
# ).encode(
#     alt.X("country",sort = "-y"),
#     alt.Y("count()"),
#     alt.Color("type", scale=alt.Scale(domain=["Movie", "TV Show"], range=["orangered", "purple"]))
# )

# st.write(selection = alt.selection_interval())
# country.add_selection(selection).encode(
#     color=alt.condition(selection, "type", alt.value("grey"))
# ) | hist.transform_filter(selection)



########## Part 2: selecting max and showcase graphs visualization and interaction
st.header("Part 2: More in-depth analysis of dataset by selecting top N number")

df_movies = load_specific_data(filtered_df, 'Movie')
df_tv = load_specific_data(filtered_df, 'TV Show')


@st.cache
def getTopColData(df, num, col):
  top_col_data = set(df[df[col] != 'No Data'][col].value_counts().nlargest(num).keys())
  return df[df[col].isin(top_col_data)]


@st.cache
def multivalueCounter(df, col):
	dicts = {}
	for line in list(df[col]):
		lines = line.split(", ")
		for i in lines:
			if i == 'No Data':
				continue
			elif i not in dicts:
				dicts[i]=1
			else:
				dicts[i]+=1
	return dicts


# subfunction to create genres
@st.cache
def get_top_genre_df(d, num):
	dicts = {}
	count = 0
	for k, v in sorted(d.items()):
		if count < num:
			dicts[k] = v
			count += 1
		else:
			dicts['Others'] = dicts.get('others', 0) + 1

	return pd.DataFrame({'listed_in': dicts.keys(), 'count': dicts.values()})


@st.cache
def create_genre_pie_chart(df, num, category, width=500, height=400):
	genre_dict = multivalueCounter(df, 'listed_in')
	top_genre_df = get_top_genre_df(genre_dict, 5)

	base_genre = alt.Chart(top_genre_df).encode(
	    theta=alt.Theta('count', stack=True),
	    color=alt.Color('listed_in', scale=alt.Scale(range=netflix_colors)),
	    tooltip=['listed_in', 'count']
    ).properties(
    	title=f'Genres distribution from top {num} {category}',
	    width=width,
	    height=height
	)

	genre_pie = base_genre.mark_arc(outerRadius=120)
	genre_text = base_genre.mark_text(radius=140, size=20).encode(text="count:N")
	return (genre_pie, genre_text)


# sub function to combine ratings into count
@st.cache
def get_new_ratings(df):
	dicts = {}
	for age in list(df['age_group']):
		if age == 'No Data':
			continue
		if age not in dicts:
			dicts[age] = 1
		else:
			dicts[age] += 1
	return dicts


@st.cache
def create_rating_pie_chart(df, num, category, width=500, height=400):
	
	rating_dict = get_new_ratings(df)
	rating_df = pd.DataFrame({'age_group': rating_dict.keys(), 'count': rating_dict.values()})

	base_genre = alt.Chart(rating_df).encode(
	    theta=alt.Theta('count', stack=True),
	    color=alt.Color('age_group', scale=alt.Scale(range=netflix_colors)),
	    tooltip=['age_group', 'count'],
    ).properties(
    	title=f'Ratings distribution from top {num} {category}',
	    width=width,
	    height=height
	)

	genre_pie = base_genre.mark_arc(outerRadius=120)
	genre_text = base_genre.mark_text(radius=140, size=20).encode(text="count:N")
	return (genre_pie, genre_text)

@st.cache
def create_release_year_chart(df, num, category, width=500, height=400):
	df_to_use_filtered_year = alt.Chart(df).mark_line().encode(
		alt.X('release_year',),
		alt.Y("count()"),
	    color=alt.Color('type', scale=alt.Scale(range=netflix_colors)),
		tooltip=['release_year', 'count()']
	).transform_filter(
		'datum.release_year > 1970'
	).properties(
		title=f'Release year distribution based on {top_num_slider} {category}',
		width=width,
		height=height
	)
	return df_to_use_filtered_year


@st.cache
def create_country_chart(df, num, category, width=500, height=400):
	df_to_use_filtered_country = alt.Chart(df).mark_bar().encode(
		alt.X('first_country', sort='y'),
		alt.Y("count()"),
		color=alt.Color('type', scale=alt.Scale(range=netflix_colors)),
		tooltip=['first_country', 'count()'] # 	    tooltip=['age_group', 'count'],
	).transform_filter(
		'datum.first_country != "No Data"'
	).properties(
		title=f'Countries distribution based on {top_num_slider} {category}',
		width=width,
		height=height
	)
	return df_to_use_filtered_country

### Unused as it is already shown in below through the radio box
dicts2=multivalueCounter(filtered_df, 'country')
countries = pd.DataFrame(pd.Series(dicts2, index=dicts2.keys()).reset_index())
countries.columns = ['Country','Records']
countries['Country']= np.where(countries['Country']=='',"None", countries['Country'])

# # slider
# n_country = st.slider(
#      'Select the number of countries to show their distribution',
#      1, len(countries), 15)

# countryDist = alt.Chart(countries).mark_bar().encode(
#     alt.X('Country', sort='-y'),
#     alt.Y('Records'),
#     alt.Color("Records:Q", scale=alt.Scale(range=netflix_colors),legend=None)
# ).properties(
#     title="Netflix country distribution",
#     width=600,
#     height=500
# ).transform_window(
#     rank='rank(Records)',
#     sort=[alt.SortField('Records', order='descending')]
# ).transform_filter(
#     (alt.datum.rank < n_country) # selection n
# )
# st.altair_chart(countryDist)

# country search function
@st.cache
def isCountry(countryString, country):
    if countryString.lower().find(country.lower())!=-1:
        return country
    else:
        return "Global"
def searchCountry(df, country):
    temp = []
    for countryLine in list(df['country']):
        temp.append(isCountry(countryLine, country))
    return temp

# make single selection default
# ref : https://github.com/streamlit/streamlit/issues/949




# selection country vs. global
countrySelectionList = countries.sort_values(by='Records', ascending=False)['Country']



# TODO: explain what the dataset encomprises of
# st.write("The sliced dataset contains {} elements ({:.1%} of total).".format(slice_labels.sum(), slice_labels.sum() / len(df)))

# 4k directors too much to be shown
# total_directors = filtered_df['director'].unique()
# directors_list = [i for i in range(len(total_directors))] # 4k plus
selection_list = ['Director', 'Country', 'Genres'] # 'Release_year'
type_list = ['All', 'Tv show', 'Movie']

st.write("Choose a catgory and type of show you are interested to find out more.")

cols = st.columns(2)
with cols[0]:
	category = st.radio('Category', selection_list).lower()
with cols[1]:
	type_selector = st.radio('Data Type', type_list)


# 'release_year': 10 # may not be helpful in the array
slider_limit = {
	'director': 30,
	'country': 127,
	'genres': 42
}


top_num_slider = st.slider(f'Please pick a number to show top N number for {category}', 0, slider_limit[category])
if top_num_slider != 0:
	df_to_use = filtered_df
	if type_selector == 'Tv show':
		df_to_use = df_tv
	elif type_selector == 'Movie':
		df_to_use = df_movies

	if category == 'director':
		df_to_use_filtered = getTopColData(df_to_use, top_num_slider, category)

		# st.write("Feel free to select a director to find out more about their work.")
		# select_brush = alt.selection_single(empty='all', fields=['director'])
		df_to_use_filtered_graph = alt.Chart(df_to_use_filtered).mark_bar().encode(
		    alt.X('count()'),
		    alt.Y('director', sort='-x'),
		    color=alt.Color('type', scale=alt.Scale(range=netflix_colors)),
		).properties(
			title=f'Top {top_num_slider} directors distribution',
			width=800,
			height=400
		)
	    # opacity=alt.condition(select_brush, alt.OpacityValue(1), alt.OpacityValue(0.7))

		# .add_selection(
		# 	select_brush
		# )
		# st.write(alt.layer(df_to_use_filtered_graph, line, data=df_to_use_filtered))
		# 1st graph to show general interactions
		st.altair_chart(df_to_use_filtered_graph)

		# 2nd graph: directors with listed_in (genre)
		cols = st.columns((.1, 3, 4, 2))

		# cols = st.columns(2)
		with cols[1]:
			genre_pie, genre_text = create_genre_pie_chart(df_to_use_filtered, top_num_slider, category)
			st.altair_chart(genre_pie + genre_text)

		# 3rd graph: directors with rating
		with cols[3]:
			rating_pie, rating_text = create_rating_pie_chart(df_to_use_filtered, top_num_slider, category)
			st.altair_chart(rating_pie + rating_text)

		# 4th graph: directors with country
		df_to_use_filtered_country = create_country_chart(df_to_use_filtered, top_num_slider, category)
		# 5th graph: directors with release year
		df_to_use_filtered_year = create_release_year_chart(df_to_use_filtered, top_num_slider, category)

		with cols[1]:
			st.altair_chart(df_to_use_filtered_country)
		with cols[3]:
			st.altair_chart(df_to_use_filtered_year)

	elif category == 'country':
		df_to_use_filtered = getTopColData(df_to_use, top_num_slider, category)

		# 1st graph: put country + count first
		df_to_use_filtered_country = alt.Chart(df_to_use_filtered).mark_bar().encode(
			alt.X('first_country', sort='y'),
			alt.Y("count()"),
		    color=alt.Color('type', scale=alt.Scale(range=netflix_colors)),
			tooltip=['count()'] # add 'director', 
		).properties(
			title=f'Country distribution based on {top_num_slider} {category}',
		    width=800,
		    height=500
		)
		st.altair_chart(df_to_use_filtered_country)
		
		select_brush = alt.selection(type='interval', encodings=['y'])
		dict_directors = multivalueCounter(df_to_use_filtered, 'director')
		# st.write(dict_directors)

		# 1st graph: country + director # skip for now
		# st.write(alt.layer(df_to_use_filtered_graph, line, data=df_to_use_filtered))

		# 2nd graph: countries with listed_in (genre)
		cols = st.columns((.1, 3, 4, 2))

		# cols = st.columns(2)
		with cols[1]:
			genre_pie, genre_text = create_genre_pie_chart(df_to_use_filtered, top_num_slider, category)
			st.altair_chart(genre_pie + genre_text)

		with cols[3]:
			rating_pie, rating_text = create_rating_pie_chart(df_to_use_filtered, top_num_slider, category)
			st.altair_chart(rating_pie + rating_text)


		# 5th graph: directors with release year
		# TODO: add multi line tooltip
		df_to_use_filtered_year = create_release_year_chart(df_to_use_filtered, top_num_slider, category, 800, 400)
		st.altair_chart(df_to_use_filtered_year)

	elif category == 'genres':
		dicts1=multivalueCounter(df_to_use, 'listed_in')
		genres = pd.DataFrame(pd.Series(dicts1, index=dicts1.keys()).reset_index())
		  # sorting genres descending order
		genres.columns = ['Genre','Records']

		genreDist = alt.Chart(genres).mark_bar().encode(
		    alt.X('Genre', sort='-y'),
		    alt.Y('Records'),
		    alt.Color("Records:Q", scale=alt.Scale(range=netflix_colors),legend=None),
		    tooltip=['Genre', 'Records']
		).properties(
		    title=f"Genres distribution based on {top_num_slider} {category}",
		    width=800,
		    height=500
		).transform_window(
		    rank='rank(Records)',
		    sort=[alt.SortField('Records', order='descending')]
		).transform_filter(
		    (alt.datum.rank < top_num_slider) # selection n
		)
		st.altair_chart(genreDist)


	### Remove release year for now
	# elif category == 'release_year':
	# 	st.write("should show release year graph")

	# 	df_to_use_filtered_year = alt.Chart(df_to_use_filtered).mark_bar().encode(
	# 		alt.X('release_year',),
	# 		alt.Y("count()"),
	# 		color='type',
	# 		tooltip='count()'
	# 	).properties(
	# 		width=800,
	# 		height=400
	# 	)
	# 	st.write(df_to_use_filtered_year)

		# 1st graph: put country + count first
		# df_to_use_filtered_country = alt.Chart(df_to_use_filtered).mark_bar().encode(
		# 	alt.X('first_country', sort='y'),
		# 	alt.Y("count()"),
		# 	color='type',
		# 	tooltip=['count()'] # add 'director', 
		# )
		# st.write(df_to_use_filtered_country) # convert to map
		
		# select_brush = alt.selection(type='interval', encodings=['y'])
		# dict_directors = multivalueCounter(df_to_use_filtered, 'director')
		# st.write(dict_directors)

		# 1st graph: country + director # skip for now
		# st.write(alt.layer(df_to_use_filtered_graph, line, data=df_to_use_filtered))

		# 2nd graph: countries with listed_in (genre)
		# genre_pie, genre_text = create_genre_pie_chart(df_to_use_filtered)
		# st.altair_chart(genre_pie + genre_text)

		# 3rd graph: countries with rating
		# rating_pie, rating_text = create_rating_pie_chart(df_to_use_filtered)
		# st.altair_chart(rating_pie + rating_text)


		# 5th graph: directors with release year


# TODO: add a random sampling of movie interaction to see more info

########## Part 3: Comparisons between 2 options visualization and interaction
st.header('Part 3: In-depth analysis on country with genre')
st.subheader("Heat map for top 10 country")

country_agegroup = filtered_df.groupby(['first_country','age_group']).size().unstack().reset_index()
country_agegroup.columns = ['country','Adults','Kids','Older Kids','Teens']
country_agegroup.fillna(0, inplace=True)
# make percentage
country_agegroup['total']=country_agegroup['Adults']+country_agegroup['Kids']+country_agegroup['Older Kids']+country_agegroup['Teens']
for c in country_agegroup.columns[1:]:
	country_agegroup[c]=(country_agegroup[c]/(country_agegroup['total'])*100).round(2)

# n country slider
n_country = st.slider(
     'Select the number of country to show their rating heatmap',
     1, 30, 10)

#  select n countries for heatmap
final_country_age = pd.melt(country_agegroup, id_vars='country', value_vars=country_agegroup.columns[1:-1])
# exchange countries
selection = countrySelectionList.head(n_country)
final_country_age.columns = ['Country','Type','Ratio']
selected_country_age = final_country_age[final_country_age.Country.isin(selection)].sort_values(by=['Country','Type'])



# show heatmap
n_width = 35*n_country
heatmap = alt.Chart(selected_country_age).mark_rect().encode(
    alt.X('Country',type="ordinal", title='Countries'),
    alt.Y('Type',type="ordinal", title='Media Type'),
    alt.Tooltip(["Country","Type","Ratio"]),
    color=alt.Color('Ratio:Q', title='Ratio',scale=alt.Scale(range=netflix_colors),legend=None)
).properties(
    title="Countries Media type's Ratio",
    width=n_width,
    height=60*4,
).configure_axis(
        labelFontSize=10,
        titleFontSize=10,
        domainWidth=0.5
    ).configure_title(
    fontSize=15,
    # anchor='start',
    color='black'
)
st.altair_chart(heatmap)



########################################
@st.cache
def isCountry2(countryString, country):
    if countryString.lower().find(country.lower())!=-1:
        return True
    else:
        return False
@st.cache
def searchCountry2(df, country):
    temp = []
    for countryLine in list(df['country']):
        temp.append(isCountry2(countryLine, country))
    return temp
@st.cache
def createCountryDf(df,country1, country2, column_selector):
	testsearchList1 = searchCountry2(df,country1)
	testsearchList2 = searchCountry2(df,country2)
	selected_column = column_selector
	if sum(testsearchList1)>sum(testsearchList2):
		country1df = pd.DataFrame(df[testsearchList1][selected_column])
		country2df = pd.DataFrame(df[testsearchList2][selected_column])
	else:
		country1df = pd.DataFrame(df[testsearchList2][selected_column])
		country2df = pd.DataFrame(df[testsearchList1][selected_column])
		country1, country2  = country2, country1
	return country1, country2, country1df, country2df

st.subheader("Comparisons between 2 countries")

countrySelection = countries.sort_values(by='Records', ascending=False)['Country']

location = st.multiselect("Select 2 countries to compare", countrySelection)
button = st.button("Show comparison chart",disabled=False)

if button :
	if len(location) == 2:
		country1 = location[0]
		country2 = location[1]
		selected_column = 'release_year' # 'year_added'
		country1, country2, df1, df2 = createCountryDf(filtered_df, country1, country2, selected_column)
		graph_title = (country1 + " vs. " + country2 + " comparison over "+ selected_column)

		g1= alt.Chart(df1).mark_area(color='#221f1f',tooltip=True).encode(
		alt.X(selected_column,title="Released Year("+country1+")", scale=alt.Scale(zero=False, padding=0), axis=alt.Axis(format = 'd', title="Released Year")),
		alt.Y("count()",title='Number of Media'),
		).properties(
		width = 600
		)
		g2= alt.Chart(df2).mark_area(color='#b20710',tooltip=True).encode(
		alt.X(selected_column,title="Released Year("+country2+")", scale=alt.Scale(zero=False, padding=0), axis=alt.Axis(format = 'd', title="Released Year")),
		alt.Y("count()",title='Number of Media'),
		).properties(
		title=graph_title,
		width = 600
		)
		st.altair_chart(g1+g2)
	else:
		st.warning("You have to select only 2 locations")

st.subheader("A country's relase year trend vs. global")
#
countrySelectionList_modi = [''] + list(countrySelectionList)
countrySingleSelection = st.selectbox('Select country:', countrySelectionList_modi)

if countrySingleSelection!='':	
    # show country graph with single selection
	countrySelector = pd.DataFrame(searchCountry(filtered_df,countrySingleSelection), index=filtered_df.index)
	countrySelector.columns = ['search']
	df_graph = pd.concat([filtered_df[['release_year']],countrySelector],axis=1)
	titleString = str(countrySingleSelection)+"'s release year record with global trend"

	showCountry = st.checkbox("Show Country Raw Data")
	if showCountry:
		st.write(df_graph)

	# hide graph (default for loading spped)
	button = st.button("Show country vs. global chart",disabled=False)
	if button :
		selectedCountry = alt.Chart(df_graph).mark_area(
			tooltip=True
		).encode(
			alt.X("release_year", scale=alt.Scale(zero=False, padding=0), axis=alt.Axis(format = 'd', title="Released Year")),
			alt.Y('count()', scale=alt.Scale(zero=False), title="Records"),
			alt.Color("search", scale=alt.Scale(range=['#221f1f','#b20710']),
					legend=alt.Legend(title="Country",)
					)
		).properties(
			width=600,
			title=titleString
		)
		st.altair_chart(selectedCountry)

