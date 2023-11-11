import folium
import streamlit as st
import pandas as pd


from streamlit_folium import folium_static



# center on Liberty Bell, add marker
m = folium.Map(location=[0,0], zoom_start=1)



csv_file_path = 'data/metadata.csv'
df = pd.read_csv(csv_file_path)
data_list = df.to_dict(orient='records')
print(data_list[0])

for item in data_list:
    lat = round(item['lat'], 3)
    lon = round(item['lon'], 3)
    if item['plume'] == "yes":
        icon=folium.Icon(color='black',icon_color='#FF0000')
    else:
        icon=folium.Icon(color='black',icon_color='#00FF00')
    name = "<b>id:</b>" + item['id_coord'].split('_')[1] + "</br>" + "<b>lat:</b>" + str(lat) + "</br>" + "<b>lon:</b>" + str(lon)
    
    folium.Marker([lat, lon], popup=name, icon=icon).add_to(m)#

# call to render Folium map in Streamlit
st_data = folium_static(m, width=725)