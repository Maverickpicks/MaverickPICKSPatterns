from data_loader import load_stock

from data_loader import load_nifty


from trend_engine_v2 import trend_analysis

from momentum_engine_v2 import momentum_analysis

from relative_strength_v2 import relative_strength_analysis


from volume_engine import volume_analysis

from risk_engine import risk_analysis

from pattern_engine import pattern_analysis


from ranking_engine_v2 import ranking_engine



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



result=ranking_engine(

    trend,

    momentum,

    rs,

    volume,

    pattern,

    risk

)



print()

print("TREND")

print(trend)



print()

print("MOMENTUM")

print(momentum)



print()

print("RS")

print(rs)



print()

print("VOLUME")

print(volume)



print()

print("PATTERN")

print(pattern)



print()

print("RISK")

print(risk)



print()

print("FINAL")

print(result)