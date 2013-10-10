/**********************************************************************
The Plot Window Widget

Copyright (c) 2013, yt Development Team.

Distributed under the terms of the Modified BSD License.

The full license is in the file COPYING.txt, distributed with this software.
***********************************************************************/


Ext.define("Reason.view.widgets.PhasePlot", {
    extend: 'Ext.panel.Panel',
    title: 'This should not be visible.',
    alias: 'widget.phaseplotwindow',
    iconCls: 'graph',
    autoScroll: true,
    layout: 'absolute',
    width: '100%',
    height: '100%',
    closable: true,

    items: [
        {
            xtype: 'panel',
            itemId: 'y_ticks',
            layout: 'absolute',
            y: 10,
            x: 100,
            width: 40,
            height: 400,
            items : [],
            border: false,
        }, {
            xtype: 'panel',
            itemId: 'x_ticks',
            layout: 'absolute',
            y: 410,
            x: 140,
            width: 400,
            height: 40,
            items : [],
            border: false,
        }, {
            xtype:'image',
            itemId: 'imagepanel',
            src: 'reason/resources/images/loading.png',
            style: 'border: 1px solid #000000;',
            x: 138,
            y: 8,
            width: 400,
            height: 400,
        }, {
            xtype:'image',
            itemId: 'colorbar',
            style: 'border: 1px solid #000000;',
            x: 560,
            y: 10,
            width: 30,
            height: 400,
        }, {
            xtype: 'panel',
            itemId: 'cb_ticks',
            layout: 'absolute',
            y: 10,
            x: 590,
            width: 40,
            height: 400,
            items : [],
            border: false,
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
            xtype: 'panel',
            layout: 'vbox',
            itemId: 'rhs_panel',
            width: 300,
            height: 460,
            x: 640, y: 10,
            layout: 'absolute',
            items: [
                {
                  xtype: 'panel',
                  title: 'Plot MetaData',
                  itemId: 'metadataString',
                  style: {fontFamily: '"Inconsolata", monospace'},
                  html: 'Welcome to the Plot Window.',
                  height: 200,
                  width: 300,
                  x: 0,
                  y: 0,
                }, {
                  xtype: 'panel',
                  title: 'Plot Editor',
                  itemId: 'plot_edit',
                  height: 460,
                  width: 300,
                  x: 0,
                  y: 200,
                },
           ],
        }
    ],
});
