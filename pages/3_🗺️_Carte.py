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




def recherche(id):
    for item in data_list:
        if str(item['id_coord']) == "id_" + id:
            col1 = ["Id", "Lattitude", "Longitude", "Plume", "X", "Y", "Dernière date"]
            date = str(item['date'])
            date = date[6:8] + "/" + date[4:6] + "/" + date[0:4]
            col2 = [item['id_coord'], item['lat'], item['lon'], item['plume'], item['coord_x'], item['coord_y'], date]
            table = pd.DataFrame({"Infos": col1, "Valeurs": col2})
            st.table(table)
            path = ""
            if item['plume']:
                path += "data/images/plume/"
            else:
                path += "data/images/no_plume/"
            path += str(item['date'])+"_methane_mixing_ratio_"+item['id_coord'] +  ".tif"
            original_image = Image.open(path)
            #img = load_img(original_image, target_size = (64,64), color_mode = "grayscale")
            plt.imshow(original_image, cmap='gray')
            plt.axis('off')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            return
    if id !="":
        st.warning("Id inexistant")
    return

st.markdown("<h1 style='text-align: center'>Rechercher un point</h1>", unsafe_allow_html=True)
with st.form("Lieux", clear_on_submit=True):
    name = st.text_input("Id du point recherché :")
    valid_status = st.form_submit_button("Rechercher", on_click=recherche(name))
    if valid_status:
        if name == "":
            st.warning("Rentrer un id")
