"""
Name: World Regions
Desc: Define dictionary lon and lat sections for the land masses
"""

regions = {
    "north_america":[],
    "south_america":[],
    "africa":[{"lon":(-18,59.75),"lat":(3,39.75)},
              {"lon":(8,45),"lat":(-15,2.75)},
              {"lon":(10,41),"lat":(-35,-15.25)},
              {"lon":(41.25,51),"lat":(-26,-11)}],
    "europe_asia":[{"lon":(-18,59.75),"lat":(3,39.75)}],
    "australia":[],
    "north_pole":[],
    "south_pole":[],
    "islands":[],
    "east_hemisphere":[{"lon":(-20,180),"lat":(-45,85)}],
    "west_hemisphere":[{"lon":(-180,0),"lat":(53,85)},
                       {"lon":(-130,-50),"lat":(10,53)},
                       {"lon":(-90,-30),"lat":(-58,10)}],
    "antartica":[{"lon":(-180,180),"lat":(-90,-62)}]
}


east_water = [{"lon":(125,180),"lat":(0,30)},
         {"lon":(150,180),"lat":(-10,0)},
         {"lon":(50,95),"lat":(-65,4)},
         {"lon":(150,180),"lat":(30,50)}]


west_water = [{"lon":(-75,-50),"lat":(20,35)},
         {"lon":(-130,-110),"lat":(10,24)}]
