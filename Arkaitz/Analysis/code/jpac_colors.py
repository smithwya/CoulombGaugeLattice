###############################################################################
#   JPAC color style
###############################################################################
jpac_blue   = "#1F77B4"; jpac_red    = "#D61D28"; jpac_green  = "#2CA02C";jpac_green3 = "#53C98C";
jpac_orange = "#FF7F0E"; jpac_purple = "#9467BD"; jpac_brown  = "#8C564B";
jpac_pink   = "#E377C2"; jpac_gold   = "#BCBD22"; jpac_aqua   = "#17BECF";
jpac_grey   = "#7F7F7F"; jpac_yellow = "#E7E823"; jpac_green2 = "#70a02c";
jpac_color = [jpac_blue, jpac_red, jpac_green, jpac_orange, jpac_purple,
              jpac_brown, jpac_pink, jpac_gold, jpac_aqua, jpac_grey, 'black' ];

jpac_color_rainbow= [jpac_pink, jpac_red, jpac_pink, jpac_brown, jpac_orange, jpac_gold, jpac_yellow, jpac_green2, jpac_green, jpac_aqua, jpac_blue, jpac_purple,
                  jpac_grey, 'black' ];
jpac_color_inv_rainbow= [jpac_purple, jpac_blue, jpac_aqua, jpac_green, jpac_green2, jpac_yellow, jpac_gold, jpac_orange, jpac_brown, jpac_red, jpac_pink,  
                  jpac_grey, 'black' ];

jpac_color_around= [jpac_purple, jpac_blue, jpac_aqua, jpac_green3, jpac_green ,  jpac_yellow, jpac_gold, jpac_orange, jpac_red, jpac_pink,  
                  jpac_grey,jpac_purple, jpac_blue, jpac_aqua, jpac_green, jpac_green2, jpac_yellow, jpac_gold, jpac_orange, jpac_brown, jpac_red, jpac_pink,  
                  jpac_grey, 'black' ];

jpac_color_around_symm= [jpac_purple, jpac_blue, jpac_aqua, jpac_green, jpac_green2, jpac_yellow, jpac_gold, jpac_orange, jpac_brown, jpac_red, jpac_pink,  
                  jpac_grey, jpac_pink, jpac_red, jpac_pink, jpac_brown, jpac_orange, jpac_gold, jpac_yellow, jpac_green2, jpac_green, jpac_aqua, jpac_blue, jpac_purple, 'black' ];

dashes, jpac_axes = 10*'-', jpac_color[10];

markerlist=["o","s","p","8","v","^","<",">","1","2","3","4"];