from data_loader import load_stock

from data_loader import load_nifty

from relative_strength_v2 import relative_strength_analysis


stock=load_stock(

    "BEL"

)


nifty=load_nifty()


result=relative_strength_analysis(

    stock["daily"],

    nifty

)


print()

print(result)