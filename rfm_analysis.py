import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import data_and_attributes as da
import modelling as model
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout = "wide")

def get_aggregation(data : pd.DataFrame) -> pd.DataFrame:
    aggregate = data.groupby(by = ['customer_id'], as_index = False).agg(lastpurchasedate = ('order_date' , pd.Series.max),
                                                                         frequency = ('order_id', pd.Series.nunique),
                                                                         monetary = ('total_price', pd.Series.sum))
    
    aggregate['recency'] = (data['order_date'].max() + dt.timedelta(1) - aggregate['lastpurchasedate']).dt.days
    aggregate = aggregate[['customer_id', 'lastpurchasedate', 'recency', 'frequency', 'monetary']]
    return(aggregate)

def modelling(data : pd.DataFrame, **set_params) -> pd.DataFrame:
    standarization = set_params.get('standarization', True)
    scalertype = set_params.get('scalertype', 'standartscaler')
    n_clusters = set_params.get('n_clusters', 5)
    iteration  = set_params.get('iteration', 300)
    
    parameters = dict(standarization = standarization,
                      scalertype = scalertype,
                      n_clusters = n_clusters,
                      iterations = iteration)
    
    fitmodel = model.Modelling(data = data[['recency', 'frequency', 'monetary']])\
                    .clustering(clustertype = 'kmeans',
                                set_params = parameters)
          
    data['cluster'] = np.array(map(str, fitmodel.labels_))
    data = data.sort_values(by = ['cluster', 'customer_id'],
                            ignore_index = True)
    return fitmodel, data
    
def header():
    teks = []
    
    spacer1, row1, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row1:
        st.image('https://raw.githubusercontent.com/bachtiyararief/rfm_kmeans/main/HeaderRFM.jpg')
        st.subheader('Streamlit App by [Bachtiyar M. Arief](https://www.linkedin.com/in/bachtiyarma/)')
        
        teks.append('Setiap pelaku usaha tentu menginginkan produknya laku keras di pasaran.\
                     Oleh karena itu, tidak heran kalau dalam upaya pemasarannya, banyak pelaku usaha\
                     yang berusaha menawarkan produk barang atau jasanya ke semua orang tanpa ada\
                     segmentasi pasar secara khusus. Harapannya, produk barang atau jasanya\
                     mendapat respons yang positif dan sanggup memenuhi kebutuhan setiap orang.\
                     Namun, apakah kondisi di lapangan berjalan seperti itu ? Pada kenyataannya,\
                     pelanggan yang bervariasi tentu memiliki karakter dan ekspektasi yang berbeda-beda pula.\
                     Disinilah pentingnya peran segmentasi pelanggan untuk dilakukan.') 
                 
        teks.append('Segmentasi pelanggan (<i>customer segmentation</i>) adalah teknik mengelompokkan pelanggan\
                     ke dalam kelompok berdasarkan pola perilaku pembelian mereka. Dalam segmentasi pelanggan,\
                     berbagai kriteria juga dapat digunakan tergantung pada kondisi pasar seperti kondisi\
                     geografis, karakteristik demografis atau basis perilaku.\
                     Teknik ini mengasumsikan bahwa grup dengan fitur yang berbeda\
                     memerlukan pendekatan pemasaran yang berbeda. Dengan demikian strategi pemasaran yang tepat\
                     dan efektif dalam bisnis dapat dikembangkan, dapat menjual - melayani - memelihara hubungan yang lebih baik\
                     dengan pelanggan serta meningkatkan profitabilitas yang efektif untuk setiap segmen bisnis.')
        
        teks.append('Analisis <b>Recency, Frequency, Monetary</b> (RFM) merupakan salah satu proses analisis segmentasi pelanggan.\
                     Dalam menentukan segmentasi pelanggan, digunakan model RFM berdasarkan tiga variabel yaitu :')
        
        for text in teks:
            formattext = da.Formater(text = text).text_markdown(align = 'justify')
            st.markdown('\n{}'.format(formattext), unsafe_allow_html = True)
        
        st.text('\t1. Recency   : Waktu interaksi terakhir pelanggan dengan produk\n2. Frequency : Berapa kali pelanggan berinteraksi dengan produk atau melakukan transaksi dalam periode waktu tertentu\n3. Monetary  : Jumlah total yang dihabiskan oleh pelanggan untuk membeli produk Anda dalam periode waktu tertentu')        
        
        teks1 = "Proses analisa segmentasi pelanggan dengan RFM adalah sebagai berikut"
        
        teks1 = da.Formater(text = teks1).text_markdown(align = 'justify')
        st.markdown('\n{}'.format(teks1), unsafe_allow_html = True)
        
def show_data(data : pd.DataFrame):
    formatdata = {'quantity' : '{:.0f}',
                  'item_price' : '{:,.2f}',
                  'total_price' : '{:,.2f}'}
    
    showdata = da.Formater(data = data).format_show_data(formats = formatdata)

    spacer1, row2, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row2:
        st.subheader('1. Data')
        teks2 = 'Data yang digunakan pada proses analisa segementasi pelanggan dengan RFM\
                 bersumber dari API <a href="https://dataset.dqlab.id/">DQLab</a> (<i>dengan <b>contents key</b> : 10%_original_randomstate=42/retail_data_from_*_until_*_reduce</i>).\
                 Dataset mempunyai format .csv yang tersebar pada link yang berbeda sehingga perlu dilakukan proses ekstraksi dan\
                 integrasi data terlebih dahulu.'
                 
        teks2 = da.Formater(text = teks2).text_markdown(align = 'justify')
        st.markdown('\n{}'.format(teks2), unsafe_allow_html = True)
        
        st.dataframe(data)
        
        kolomdesc = '\n1.  order_id\t: ID dari order atau transaksi, 1 transaksi bisa terdiri dari beberapa produk, tetapi hanya dilakukan oleh 1 customer\
                     \n2.  order_date\t: tanggal terjadinya transaksi\
                     \n3.  customer_id\t: ID dari pembeli; bisa jadi dalam satu hari, 1 customer melakukan transaksi beberapa kali\
                     \n4.  city\t: kota tempat toko terjadinya transaksi\
                     \n5.  province\t: provinsi (berdasarkan city)\
                     \n6.  product_id\t: ID dari suatu product yang dibeli\
                     \n7.  brand\t: brand/merk dari product. Suatu product yang sama pasti memiliki brand yang sama\
                     \n8.  quantity\t: Kuantitas/banyaknya product yang dibeli\
                     \n9.  item_price\t: Harga dari 1 product (dalam Rupiah). Suatu product yang sama, bisa jadi memiliki harga yang berbeda saat dibeli\
                     \n10. total_price\t: Hasil kali barang dibeli (quantity) dengan harga barang (item_price)'
        
        st.text("\n Deskripsi Kolom\n" + kolomdesc)

def show_aggregation(data : pd.DataFrame):
    spacer1, row3_1, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row3_1:
        st.subheader('2. Aggregasi')
    
    spacer1, row3_2, spacer2, row3_3, spacer3 = st.columns((0.1, 4.2, 0.1, 4.2, 0.1))
    with row3_2:
        teks3 = 'Berdasarkan analisa RFM perlu dilakukan aggregasi untuk mendapatkan\
                 nilai <i>recency - frequency - monetary</i> pada data.</br>\
                 <br>a. <b>Recency</b> : Hitung jarak hari antara <i>latest</i> order_date + 1 hari dengan tanggal pembelian terakhir tiap customer_id</br>\
                 <br>b. <b>Frequency</b> : Hitung jumlah order_id yang unik (<i>count distinct</i>) pada tiap customer_id</br>\
                 <br>c. <b>Monetary</b> : Hitung total pengeluaran (<i>sum of total_price</i>) tiap customer_id</br>'
                 
        teks3 = da.Formater(text = teks3).text_markdown(align = 'justify')
        st.markdown('\n{}'.format(teks3), unsafe_allow_html = True)
    
    with row3_3: 
        st.dataframe(data)        

def overview(data : pd.DataFrame, indicator : str, histcolor : str):
    
    indicator = indicator.lower()
    
    spacer1, row4, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row4: 
        st.subheader("Indikator " + indicator.capitalize())
        
    spacer1, row4_1, spacer2, row4_2, spacer3, row4_3, spacer4, row4_4, spacer5, row4_5, spacer6 = st.columns((0.7, 2, 0.1, 2, 0.1, 2, 0.1, 2, 0.1, 2, 0.4))
    row4_1.metric("Rataan", round(data[indicator].mean(), 2))
    row4_2.metric("Simpangan Baku", round(data[indicator].std(),2))
    row4_3.metric("Minimum", round(data[indicator].min(),2))
    row4_4.metric("Maksimum", round(data[indicator].max(),2))
    row4_5.metric("Modus", round(data[indicator].mode(),2))
    
    fig1 = px.histogram(data, x = indicator.lower(),
                       color_discrete_sequence=[histcolor],
                       title = 'Histogram ' + indicator.capitalize()) 

    fig2 = go.Figure(data=[go.Box(x = data[indicator], 
                                  boxpoints = 'outliers',
                                  marker_color = histcolor,
                                  name = indicator)]
                    )

    fig2.update_yaxes(visible = False)
    fig2.update_layout(title_text = 'Boxplot ' + indicator.capitalize())
    
    spacer1, row4_6, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row4_6:      
        st.plotly_chart(fig1, use_container_width = True)
        st.plotly_chart(fig2, use_container_width = True)
    
def show_dataoverview(data : pd.DataFrame):
    
    korelasi = data.corr(method ='pearson')
    fig = px.imshow(korelasi,
                    title = 'Korelasi R-F-M',
                    text_auto = True,
                    aspect = "auto",
                    color_continuous_scale = 'peach')\
             .update_xaxes(side="top")
             
    spacer1, row4a_1, spacer2, row4a_2, spacer3 = st.columns((0.1, 4, 0.1, 4, 0.1))
    with row4a_1:
        st.subheader('3. Data Overview')
        
        teks1 = 'Perlu dilakukan analisa dasar terlebih dahulu dengan tujuan untuk lebih mengenal data yang ada \
                 yaitu mengetahui tentang informasi dasar dari variabel yang ada di dalam data serta menonjolkan\
                 variabel - variabel yang saling berhubungan atau berkorelasi. Analisa dasar pada bagian ini terdiri dari\
                 Perhitungan <b>matriks korelasi</b> (untuk menyatakan ada atau tidaknya hubungan antar variabel dan\
                 besarnya sumbangan variabel satu terhadap yang lainnya yang dinyatakan dalam persen), \
                 perhitungan <b>statistika dasar</b> (rataan, minimum, maksimum, dll), <b>plot histogram</b> (untuk untuk mengetahui\
                 distribusi atau penyebaran suatu data) dan <b>boxplot</b> (untuk memahami karakteristik dari distribusi data \
                 yang dapat dilihat dari tinggi atau panjang boxplot juga dapat digunakan untuk menilai kesimetrisan sebaran data\
                 dan pencilan (<i>outliers</i>) yaitu nilai-nilai yang berada diatas atau dibawah ambang batas penyebarannya)'
        
        teks1 = da.Formater(text = teks1).text_markdown(align = 'justify')
        st.markdown('\n{}'.format(teks1), unsafe_allow_html = True)
    
    with row4a_2:   
        colormap = dict(Recency = 'orange',
                        Frequency = 'indianred',
                        Monetary = 'purple')
        
        st.markdown('')
        st.markdown('')
        st.plotly_chart(fig, use_container_width = True)
    
    spacer1, row4c_1, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row4c_1:
        selectindicator = st.multiselect('Pilih indikator',
                                         list(colormap.keys()),
                                         default = list(colormap.keys()))
        
    for indicator in selectindicator:
        overview(data, indicator, histcolor = colormap[indicator])

def cluster_category(centroid : pd.DataFrame):
    
    centroid['R'] = centroid['R'].max() - centroid['R'] + 1
    
    #Scoring
    for i in ['R', 'F', 'M']:
        centroid[i + '_Score'] = pd.qcut(centroid[i], 5, labels = range(1, 6))
    
    conditions = [
            (centroid['R_Score'].between(4, 5)) & (centroid['F_Score'].between(4, 5)) & (centroid['M_Score'].between(4, 5)),
            (centroid['R_Score'].between(2, 4)) & (centroid['F_Score'].between(3, 4)) & (centroid['M_Score'].between(4, 5)),
            (centroid['R_Score'].between(3, 5)) & (centroid['F_Score'].between(1, 3)) & (centroid['M_Score'].between(1, 3)),
            (centroid['R_Score'].between(4, 5)) & (centroid['F_Score'] < 2) & (centroid['M_Score'] < 2),
            (centroid['R_Score'].between(3, 4)) & (centroid['F_Score'] < 2) & (centroid['M_Score'] < 2),
            (centroid['R_Score'].between(3, 4)) & (centroid['F_Score'].between(3, 4)) & (centroid['M_Score'].between(3, 4)),
            (centroid['R_Score'].between(2, 3)) & (centroid['F_Score'] < 3) & (centroid['M_Score'] < 3),
            (centroid['R_Score'] < 3) & (centroid['F_Score'].between(2, 5)) & (centroid['M_Score'].between(2, 5)),
            (centroid['R_Score'] < 2) & (centroid['F_Score'].between(4, 5)) & (centroid['M_Score'].between(4, 5)),
            (centroid['R_Score'].between(2, 3)) & (centroid['F_Score'].between(2, 3)) & (centroid['M_Score'].between(2, 3))
        ]
    choices = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'New Customers',
               'Promising', 'Need Attention', 'About to Sleep', 'At Risk',
               'Cant Lose Them', 'Hibernating']
    
    centroid['cluster_category'] = np.select(conditions, choices, default = 'Lost')
    centroid['cluster'] = list(map(lambda ls: 'Cluster ' + str(ls), centroid.index + 1))
    return centroid

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')
        
def show_modelling(data : pd.DataFrame):
    spacer1, row4_1, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row4_1:
        st.subheader('4. K-Means Clustering')
        teks1 = 'Pengelompokan (<i>clustering</i>) merupakan salah satu analisis\
                 yang digunakan untuk mengidentifikasi cluster pada data yang memiliki\
                 kemiripan data satu dengan yang lainnya. Tujuan dari <i>clustering</i>\
                 antara lain adalah untuk menyortir objek yang berbeda dalam kelompok\
                 yang derajat asosiasinya antar dua objek akan lebih maksimal jika kedua\
                 object tersebut berada dalam satu kelompok yang sama, jika tidak, maka nilanya akan minimal.\
                 \n\nSalah satu algoritma <i>clustering</i> adalah <b>K-Means</b>. Algoritma K-Means\
                 merupakan salah satu yang paling sederhana dan mudah untuk dilakukan,\
                 relative cepat, serta mudah beradaptasi. Konsep dari K-Means adalah\
                 menempatkan setiap obyek ke dalam klaster yang terdekat dengan <i>centroid</i> (means).'
        
        teks1 = da.Formater(text = teks1).text_markdown(align = 'justify')
        st.markdown('\n{}'.format(teks1), unsafe_allow_html = True)
               
    spacer1, row4_2, spacer2, row4_3, spacer3, row4_4, spacer4 = st.columns((0.1, 2, 0.1, 3, 0.1, 2, 0.1))
    with row4_2:
        
        st.markdown('')
        st.markdown('')
        st.markdown('Input Parameter K-Means')
        
        n_cluster = st.number_input('Banyak Cluster',
                                    min_value = 1,
                                    max_value = 15,
                                    value = 5)
        
        iteration = st.slider('Maksimum Iterasi', 1, 1000, 300)
        
        scaler = st.radio("Pilih Metode Scaler", ('Standard Scaler',
                                                  'Min-Max Scaler',
                                                  'Maximum Absolute Scaler',
                                                  'Robust Scaler'))
        
        fit_rfm, result_rfm = modelling(data, 
                                        standarization = True,
                                        scalertype = scaler,
                                        n_clusters = n_cluster,
                                        iteration = iteration)
    
    plotcolor = px.colors.qualitative.Light24[0:n_cluster]
    
    result_rfm['cluster'] = result_rfm['cluster'].apply(lambda x: 'Cluster ' + str(int(x) + 1))
        
    fig1 = px.scatter_3d(result_rfm,
                         x = 'recency', 
                         y = 'frequency', 
                         z = 'monetary',
                         title = 'Hasil Clustering',
                         color = 'cluster',
                         color_discrete_sequence = plotcolor,
                         hover_data=['customer_id'])
        
    with row4_3:
        st.plotly_chart(fig1, use_container_width = True)
            
    labels = result_rfm.groupby('cluster', as_index = False).agg(total_customer = ('cluster', pd.Series.count))

    fig3 = go.Figure(data=[go.Pie(labels = labels['cluster'],
                                  values = labels['total_customer'],
                                  hole = 0.6,
                                  textinfo = 'label+percent',
                                  insidetextorientation = 'radial',
                                  showlegend = False)])
    
    fig3.update_traces(marker = dict(colors = plotcolor))
    fig3.update_layout(title_text = 'Presentase Anggota Cluster')
    
    with row4_4:
        st.plotly_chart(fig3, use_container_width = True)

    
    spacer1, row4_5, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row4_5:
        st.subheader('5. Analisa Hasil Model')
        
        teks1 = 'Setelah perhitungan algoritma K-Means dilakukan, langkah selanjutnya adalah\
                 memberi label tiap cluster. Sebelum melakukan <i>labelling</i> perlu dilakukan scoring\
                 R - F - M pada centroid (pusat cluster yang menjadi patokan pengelompokan) dimulai dari angka 1.\
                 untuk <i>score</i> terendah dan 5 untuk <i>score</i> tertinggi. Semakin tinggi nilai <i>score</i>-nya\
                 maka akan semakin baik hasilnya begitu pula sebaliknya.'
                 
        teks1 = da.Formater(text = teks1).text_markdown(align = 'justify')
        st.markdown('\n{}'.format(teks1), unsafe_allow_html = True) 
    
    cluster_center = pd.DataFrame(fit_rfm.cluster_centers_,
                                  columns = ['R', 'F', 'M'])

    cluster_center = cluster_category(cluster_center)
    cluster_center = cluster_center.merge(labels, on = ['cluster'], how = 'inner')
    cluster_center['color'] = plotcolor
    
    cluster_center_transpose = cluster_center[['R', 'F' ,'M']].T
    cluster_center_transpose.rename(columns = lambda x: 'Cluster ' + str(int(x) + 1), inplace=True)
    
    fig2 = px.line(cluster_center_transpose,
                   title = 'Cluster Type',
                   color_discrete_sequence = cluster_center['color'],
                   markers = True)
    
    fig2.update_layout({'plot_bgcolor' : 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
                        xaxis = dict(showgrid = False),
                        yaxis = dict(showgrid = False))
    
    fig2.update_yaxes(visible = False)
    
    spacer1, row4_6, spacer2, row4_7, spacer3 = st.columns((0.1, 4, 0.1, 7, 0.1))
    with row4_6:
        st.markdown('')
        st.markdown('')
        st.markdown('Scoring R-F-M')
        st.markdown('')
        st.markdown('')
        st.dataframe(cluster_center[['cluster', 'R_Score', 'F_Score', 'M_Score']])
    with row4_7:  
        st.plotly_chart(fig2, use_container_width = True)
    
    spacer1, row4_8, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row4_8:
        teks2 = 'Setelah proses <i>scoring</i>, selanjutnya adalah pemberian label pada tiap - tiap cluster.\
                 Pemberian label standar berdasarkan kriteria pada gambar berikut :'
                 
        teks2 = da.Formater(text = teks2).text_markdown(align = 'justify')
        st.markdown('\n{}'.format(teks2), unsafe_allow_html = True) 
        
    spacer1, row4_9, spacer2 = st.columns((2.6, 7.2, 2))
    with row4_9:
        st.image('https://i0.wp.com/blog.rsquaredacademy.com/img/rfm_segments_table.png?w=450&ssl=1',
                 caption = 'Segmentation R-F-M (Sumber : https://bit.ly/3LYyw8X)',
                 width = 600)
    
    spacer1, row4_10, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row4_10:
        teks3 = 'Dengan mengacu pada ketentuan tersebut, diperoleh hasil sebagai berikut : '
                 
        teks3 = da.Formater(text = teks3).text_markdown(align = 'justify')
        st.markdown('\n{}'.format(teks3), unsafe_allow_html = True) 
    
    spacer1, row4_11, spacer2, row4_12, spacer3 = st.columns((0.1, 4, 0.1, 4, 0.1))
    with row4_11:
        
        st.markdown('')
        st.markdown('')
        st.markdown('Pemberian Label pada Cluster')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.dataframe(cluster_center[['cluster', 'R_Score', 'F_Score', 'M_Score', 'cluster_category', 'total_customer']])
        
        cluster_center = cluster_center.sort_values(by = ['total_customer'], ignore_index = True)
        fig4 = go.Figure(go.Bar(
                    x = cluster_center['total_customer'],
                    y = cluster_center['cluster'],
                    orientation = 'h'))
            
        fig4.update_traces(marker = dict(color = cluster_center['color']))
        fig4.update_layout(title_text = 'Total Customer tiap Cluster')
        
    with row4_12:
        st.plotly_chart(fig4, use_container_width = False)
    
    spacer1, row4_13, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row4_13:
        teks4 = 'Dikutip dari <i>https://www.moengage.com/blog/rfm-analysis-using-rfm-segments/</i>\
                 perlakuan pelaku usaha terhadap customer berdasarkan label segmentasi yang telah diberikan\
                 tentunya berbeda antara satu segmen dengan segmen yang lain. Strategi yang mungkin dapat\
                 dilakukan oleh pelaku usaha dalam men-treatment customernya dicontohkan pada gambar dibawah ini.\
                 Sehingga diharapkan dengan strategi yang tepat dengan pengambilan keputusan yang bijak \
                 akan mendatangkan keuntungan bagi kedua belah pihak'
                 
        teks4 = da.Formater(text = teks4).text_markdown(align = 'justify')
        st.markdown('\n{}'.format(teks4), unsafe_allow_html = True) 
    
    spacer1, row4_14, spacer2, row4_15, spacer3 = st.columns((0.1, 4, 0.1, 4, 0.1))
    with row4_14:
        st.image('https://cdn-clalk.nitrocdn.com/KqmKVeLhgFAzHWrUbBzmAbRgoFMrOqoq/assets/static/optimized/rev-8f38008/wp-content/uploads/RFM-labels-main-customer-segments-2.jpg',
                 caption = 'R-F-M Strategy (Sumber : https://bit.ly/3z42IN1)',
                 width = 500)
    with row4_15:   
        result_rfm = cluster_center[['cluster', 'cluster_category']].merge(result_rfm,
                                                                           on = ['cluster'], 
                                                                           how = 'inner')
        result_rfm = result_rfm.sort_values(by = ['cluster'], ignore_index = True)
        st.dataframe(result_rfm, height = 500)
        
        st.markdown('')
        
        st.download_button(
            label = "Download data as CSV",
            data = convert_df(result_rfm),
            file_name = 'Customer Segmentation.csv',
            mime = 'text/csv',
        )
        
    spacer1, row4_16, spacer2 = st.columns((0.1, 7.2, 0.1))
    with row4_16:
        st.subheader('6. Kesimpulan')
        teks5 = 'Marketing yang cerdas memahami pentingnya "kenali klien atau customer Anda".\
                 Daripada memeriksa seluruh basis klien secara keseluruhan, lebih baik untuk\
                 mengelompokkan mereka ke dalam kelompok-kelompok tertentu, memahami kualitas setiap pertemuan,\
                 dan terlibat di dalamnya dengan kesepakatan yang relevan. Salah satu strategi divisi yang\
                 paling terkenal, mudah digunakan, dan sukses untuk memberdayakan marketing\
                 dengan memetakan perilaku customer adalah RFM dengan segmentasi Algoritma Clustering K-Means.'
        
        teks5 = da.Formater(text = teks5).text_markdown(align = 'justify')
        st.markdown('\n{}'.format(teks5), unsafe_allow_html = True) 
        
if __name__ == "__main__":
    
    header()
    
    # Get data
    datasource = da.DataSource()
    dataclean  = datasource.get_data()
    
    show_data(dataclean)
    
    #filter on the column that is used only
    data = dataclean[['order_id', 'customer_id', 'order_date', 'total_price']]
    
    rfm_per_cust = get_aggregation(data)
    show_aggregation(rfm_per_cust)
    
    show_dataoverview(rfm_per_cust)
    
    # Modelling process 
    show_modelling(rfm_per_cust)
    

    

