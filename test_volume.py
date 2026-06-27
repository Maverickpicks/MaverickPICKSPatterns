from data_loader import load_stock

from volume_engine import volume_analysis


data=load_stock(

    "BEL"

)


result=volume_analysis(

    data["daily"],

    data["weekly"],

    data["monthly"]

)


print()

print(result)