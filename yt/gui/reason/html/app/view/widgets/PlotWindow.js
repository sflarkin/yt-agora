/**********************************************************************
The Plot Window Widget View

Copyright (c) 2013, yt Development Team.

Distributed under the terms of the Modified BSD License.

The full license is in the file COPYING.txt, distributed with this software.
***********************************************************************/

Ext.define("Reason.view.widgets.PlotWindow", {
    extend: 'Ext.panel.Panel',
    title: 'This should not be visible.',
    alias: 'widget.plotwindow',
    iconCls: 'graph',
    autoScroll: true,
    layout: 'absolute',
    width: '100%',
    height: '100%',
    closable: true,

    items: [ 
        { xtype:'image',
          itemId: 'imagepanel',
          src: 'reason/resources/images/loading.png',
          x: 100,
          y: 10,
          width: 400,
          height: 400,
          draggable: false,
        }, {
            xtype:'image',
            itemId: 'colorbar',
            src: 'reason/resources/images/loading.png',
            style: 'border: 1px solid #000000;',
            x: 510,
            y: 10,
            width: 30,
            height: 400,
        }, {
            xtype: 'panel',
            itemId: 'ticks',
            layout: 'absolute',
            y: 0,
            x: 540,
            width: 100,
            height: 420,
            items : [],
            border: false,
        }, {   xtype: 'multislider',
            itemId: 'zoomSlider',
            minValue: 0,
            maxValue: 100,
            increment: 1,
            x: 100, y: 410,
            width: 400,
        },{
            xtype: 'combo',
            text: 'Field',
            itemId: 'fieldSelector',
            x: 100,
            y: 435,
            width: 400,
            queryMode: 'local',
            editable: false,
            triggerAction: 'all',
            validateOnBlur: false,
        }, {
        /* the single buttons for 10% pan*/
            xtype:'button',
            iconCls: 'singleuparrow',
            itemId: 'singleuparrow',
            //text: 'North',
            x: 40,
            y: 10,
        }, {
            xtype:'button',
            iconCls: 'singlerightarrow',
            itemId: 'singlerightarrow',
            //text:'East',
            x : 60,
            y : 30,
        }, {
            xtype:'button',
            iconCls: 'singledownarrow',
            itemId: 'singledownarrow',
            //text: 'South',
            x: 40,
            y: 50,
        }, {
            xtype: 'button',
            iconCls: 'singleleftarrow',
            itemId: 'singleleftarrow',
            //text: 'West',
            x: 20,
            y: 30,
        }, 
        /* the double buttons for 50% pan*/
        {
            xtype:'button',
            iconCls: 'doubleuparrow',
            itemId:'doubleuparrow',
            //text: 'North',
            x: 40,
            y: 80,
        }, {
            xtype:'button',
            iconCls: 'doublerightarrow',
            itemId:'doublerightarrow',
            //text:'East',
            x : 60,
            y : 100,
        }, {
            xtype:'button',
            iconCls: 'doubledownarrow',
            //text: 'South',
            itemId: 'doubledownarrow',
            x: 40,
            y: 120,
        }, {
            xtype: 'button',
            iconCls: 'doubleleftarrow',
            itemId: 'doubleleftarrow',
            //text: 'West',
            x: 20,
            y: 100,
        },
        /* Now the zoom buttons */
        {
            xtype: 'button',
            text: 'Zoom In 10x',
            itemId: "zoomin10x",
            x: 10,
            y: 160,
            width: 80,
        },{
            xtype: 'button',
            text: 'Zoom In 2x',
            itemId: "zoomin2x",
            x: 10,
            y: 185,
            width: 80,
        },{
            xtype: 'button',
            text: 'Zoom Out 2x',
            itemId:'zoomout2x',
            x: 10,
            y: 210,
            width: 80,
        },{
            xtype: 'button',
            text: 'Zoom Out 10x',
            itemId:'zoomout10x',
            x: 10,
            y: 235,
            width: 80,
        },{
            xtype: 'button',
            text: 'Upload Image',
            itemId: 'uploadimage',
            x: 10,
            y: 285,
            width: 80,
            tooltip: "Upload the current image to " +
                     "<a href='http://imgur.com'>imgur.com</a>",
        },{
            xtype: 'button',
            text: 'Pannable Map',
            itemId: 'pannablemap',
            x: 10,
            y: 335,
            width: 80,
            tooltip: "Open a pannable map in a new tab",
        },{
            xtype: 'button',
            text: 'Multicast',
            itemId: 'multicast',
            x: 10,
            y: 385,
            width: 80,
            tooltip: "Share this session across the Internet",
        },{
            xtype: 'panel',
            itemId: 'rhs_panel',
            width: 300,
            height: 460,
            x: 590, y: 10,
            layout: {
                type: 'vbox',
                align:'stretch',
                pack:'start',
            },
            items: [
                {
                  xtype: 'panel',
                  title: 'Plot MetaData',
                  itemId: 'metadataString',
                  style: {fontFamily: '"Inconsolata", monospace'},
                  html: 'Welcome to the Plot Window.',
                  height: 200,
                }, {
                  xtype: 'tabpanel',
                  itemId: 'editor_panel',
                  flex: 1.0,
                  activeTab: 0,
                  items: [
                {
                  xtype: 'panel',
                  title: 'Plot Editor',
                  itemId: 'plot_edit',
                  style: {fontFamily: '"Inconsolata", monospace'},
                  layout: 'absolute',
                  flex: 1,
                  items : [
                     {
                       x: 10,
                       y: 20,
                       width: 70,
                       xtype: 'label',
                       text: 'Display',
                     },
                     {
                       x: 80,
                       y: 20,
                       width : 80,
                       xtype: 'combo',
                       itemId: 'transform',
                       editable: false,
                       triggerAction: 'all',
                       validateOnBlur: false,
                       store: ['log10', 'linear'],
                       value: 'log10',
                     },
                     {
                       x: 10,
                       y: 60,
                       width: 70,
                       xtype: 'label',
                       text: 'Colormap',
                     },
                     {
                       x: 80,
                       y: 60,
                       width : 140,
                       xtype: 'combo',
                       itemId: 'colormap',
                       editable: false,
                       triggerAction: 'all',
                       validateOnBlur: false,
                       store: ['algae', 'RdBu', 'gist_stern',  
                               'hot', 'jet', 'kamae', 
                                'B-W LINEAR', 'BLUE',
                                'GRN-RED-BLU-WHT', 'RED TEMPERATURE',
                                'BLUE', 'STD GAMMA-II', 'PRISM',
                                'RED-PURPLE', 'GREEN', 'GRN',
                                'GREEN-PINK', 'BLUE-RED', '16 LEVEL',
                                'RAINBOW', 'STEPS', 'STERN SPECIAL',
                                'Haze', 'Blue - Pastel - Red',
                                'Pastels', 'Hue Sat Lightness 1',
                                'Hue Sat Lightness 2', 'Hue Sat Value 1',
                                'Hue Sat Value 2', 'Purple-Red + Stripes',
                                'Beach', 'Mac Style', 'Eos A', 'Eos B',
                                'Hardcandy', 'Nature', 'Ocean', 'Peppermint',
                                'Plasma', 'Blue-Red', 'Rainbow', 'Blue Waves',
                                'Volcano', 'Waves', 'Rainbow18',
                                'Rainbow + white', 'Rainbow + black'],
                       value: 'algae',
                     }
                  ]
                }, {
                  xtype: 'panel',
                  title: 'Contours',
                  itemId: 'contour_edit',
                  style: {fontFamily: '"Inconsolata", monospace'},
                  layout: 'absolute',
                  flex: 1,
                  items : [
                     {
                       x: 10,
                       y: 20,
                       width: 70,
                       xtype: 'label',
                       text: 'Field',
                     }, {
                       x: 80,
                       y: 20,
                       width : 160,
                       queryMode: 'local',
                       xtype: 'combo',
                       editable: false,
                       itemId: 'contourfield',
                       triggerAction: 'all',
                       validateOnBlur: false,
                     }, {
                       x: 10,
                       y: 60,
                       width: 70,
                       xtype: 'label',
                       text: 'Levels',
                     }, {
                       x: 80,
                       y: 60,
                       width : 160,
                       xtype: 'slider',
                       itemId: 'ncont',
                       minValue: 0,
                       maxValue: 10,
                       value: 5,
                       increment: 1,
                       /*plugins: [ {ptype: 'slidertip'} ],*/
                     }, {
                       x: 10,
                       y: 100,
                       width: 70,
                       xtype: 'label',
                       text: 'Logspaced',
                     }, {
                       x: 80,
                       y: 100,
                       width : 160,
                       xtype: 'checkbox',
                       itemId: 'logit',
                       checked: true,
                     }, {
                       x: 10,
                       y: 180,
                       width: 80,
                       xtype: 'button',
                       itemId: 'contourapply',
                       text: 'Apply',
                     }
                  ]
                }, {
                  xtype: 'panel',
                  title: 'Velocity Vectors',
                  itemId: 'vector_edit',
                  style: {fontFamily: '"Inconsolata", monospace'},
                  layout: 'absolute',
                  flex: 1,
                  items : [
                     {
                       x: 10,
                       y: 60,
                       width: 70,
                       xtype: 'label',
                       text: 'Skip Factor',
                     }, {
                       x: 80,
                       y: 60,
                       width : 160,
                       xtype: 'slider',
                       itemId: 'skip',
                       minValue: 1,
                       maxValue: 64,
                       value: 32,
                       increment: 1,
                       /*plugins: [ {ptype: 'slidertip'} ],*/
                     }, {
                       x: 10,
                       y: 180,
                       width: 80,
                       xtype: 'button',
                       itemId: 'vectorsapply',
                       text: 'Apply',
                     }
                  ]
                }
                ] } /* tabpanel items and entry */
                ]
        }
    ],
});

