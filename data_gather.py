from bing_image_downloader import downloader


list_of_species = ["Parnassians (Parnassiinae)","Swallowtails (Papilioninae)"
                    ,"Whites (Pierinae)","Sulphurs (Coliadinae)","Coppers (Lycaeninae)",
                   "Hairstreaks and Elfins (Theclinae)","Blues (Polyommatinae)","Metalmarks (Riodinidae)",
                   "Snouts (Libytheinae)","Milkweed Butterflies (Danainae)","Longwings (Heliconiinae)",
                   "True Brushfoots (Nymphalinae)","Admirals and Relatives (Limenitidinae)","Leafwings (Charaxinae)",
                   "Emperors (Apaturinae)","Satyrs and Wood-Nymphs (Satyrinae)","Spread-wing Skippers (Pyrginae)",
                   "Grass Skippers (Hesperiinae)"]
output_dir = "C:/Users/Johnny/PycharmProjects/Image_Classification/butterfly_pictures"
for i in list_of_species:
    pictures = downloader.download(query=f"{i} Butterfly",limit=100,output_dir=output_dir)

print("Printed All pictures successfully")