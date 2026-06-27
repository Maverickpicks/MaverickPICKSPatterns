from data_loader import load_stock

from data_loader import load_nifty


from trend_engine_v2 import trend_analysis

from momentum_engine_v2 import momentum_analysis

from relative_strength_v2 import relative_strength_analysis

from volume_engine import volume_analysis

from risk_engine import risk_analysis

from pattern_engine import pattern_analysis

from ranking_engine_v2 import ranking_engine

from reason_engine_v2 import generate_reason



stock=load_stock(

    "BEL"

)


nifty=load_nifty()



trend=trend_analysis(

    stock["daily"]

)


momentum=momentum_analysis(

    stock["daily"]

)


rs=relative_strength_analysis(

    stock["daily"],

    nifty

)


volume=volume_analysis(

    stock["daily"],

    stock["weekly"],

    stock["monthly"]

)


risk=risk_analysis(

    stock["daily"]

)


pattern=pattern_analysis(

    stock["daily"]

)


ranking=ranking_engine(

    trend,

    momentum,

    rs,

    volume,

    pattern,

    risk

)


result=generate_reason(

    trend,

    momentum,

    rs,

    volume,

    pattern,

    risk,

    ranking

)



print("\nTREND")
print(trend)

print("\nMOMENTUM")
print(momentum)

print("\nRS")
print(rs)

print("\nRANKING")
print(ranking)

print("\nREASON")
print(result)