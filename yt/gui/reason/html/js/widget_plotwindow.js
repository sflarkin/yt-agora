/**********************************************************************
The Plot Window Widget

Author: Cameron Hummels <chummels@gmail.com>
Affiliation: Columbia
Author: Jeffrey S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Author: Britton Smith <brittonsmith@gmail.com>
Affiliation: MSU
Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Homepage: http://yt-project.org/
License:
  Copyright (C) 2011 Matthew Turk.  All Rights Reserved.

  This file is part of yt.

  yt is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
***********************************************************************/



var WidgetPlotWindow = function(python_varname, widget_data) {
    this.id = python_varname;
    this.widget_data = widget_data;
    this.print_python = function(b, e) {
        yt_rpc.ExtDirectREPL.execute(
            {code:'print "' + python_varname + '"',
             hide:true},
            function(f, a) {alert(a.result['output']);}
        );
    }

    this.widget_keys = new Ext.KeyMap(document, [
        {key: 'z',
         shift: false,
         fn: function(){
               control_panel.get("zoom10x").handler();
            }
        },
        {key: 'Z',
         shift: true,
         fn: function(){
               control_panel.get("zoom2x").handler();
            }
        },
        {key: 'x',
         shift: false,
         fn: function(){
               control_panel.get("zoomout10x").handler();
            }
        },
        {key: 'X',
         shift: true,
         fn: function(){
               control_panel.get("zoomout2x").handler();
            }
        },
        {key: 'k',
         shift: false,
         fn: function(){
               control_panel.get("singleuparrow").handler();
            }
        },
        {key: 'j',
         shift: false,
         fn: function(){
               control_panel.get("singledownarrow").handler();
            }
        },
        {key: 'h',
         shift: false,
         fn: function(){
               control_panel.get("singleleftarrow").handler();
            }
        },
        {key: 'l',
         shift: false,
         fn: function(){
               control_panel.get("singlerightarrow").handler();
            }
        },
        {key: 'K',
         shift: true,
         fn: function(){
               control_panel.get("doubleuparrow").handler();
            }
        },
        {key: 'J',
         shift: true,
         fn: function(){
               control_panel.get("doubledownarrow").handler();
            }
        },
        {key: 'H',
         shift: true,
         fn: function(){
               control_panel.get("doubleleftarrow").handler();
            }
        },
        {key: 'L',
         shift: true,
         fn: function(){
               control_panel.get("doublerightarrow").handler();
            }
        },
    ]);
    var widget_keys = this.widget_keys;
    widget_keys.disable();
    widget_keys.varname = python_varname;

    viewport.get("center-panel").add(
        {
            xtype: 'panel',
            id: "pw_" + this.id,
            title: widget_data['title'],
            iconCls: 'graph',
            autoScroll: true,
            layout:'absolute',
            closable: true,
            listeners: {activate: function(p){
                                widget_keys.enable();
                            },
                        deactivate: function(p){
                                widget_keys.disable();
                            }
                        },
            items: [ 
                {
                    xtype:'panel',
                    id: 'image_panel_' + this.id,
                    autoEl: {
                        tag: 'img',
                        id: "img_" + this.id,
                        width: 400,
                        height: 400,
                    },
                    x: 100,
                    y: 10,
                    width: 400,
                    height: 400,
                    listeners: {
                        afterrender: function(c){
                            c.el.on('click', function(e){
                                if (e.ctrlKey == false) return;
                                xy = e.getXY();
                                cc = python_varname + ".image_recenter(" 
                                    + (xy[0] - c.el.dom.x) + ", "
                                    + (xy[1] - c.el.dom.y) + ", "
                                    + c.el.dom.width + ", "
                                    + c.el.dom.height + ")";
                                yt_rpc.ExtDirectREPL.execute(
                                {code:cc, hide:true}, cell_finished); 
                            });
                        }
                    }
                }, {
                    xtype:'panel',
                    id: 'colorbar_' + python_varname,
                    autoEl: {
                        tag: 'img',
                        id: "cb_" + python_varname,
                        src: "data:image/png;base64," +
                             widget_data['colorbar'],
                        width: 28,
                        height: 398,
                        style: 'border: 1px solid #000000;',
                    },
                    x: 510,
                    y: 10,
                    width: 30,
                    height: 400,
                }, {
                    xtype: 'panel',
                    id: 'ticks_' + python_varname,
                    layout: 'absolute',
                    y: 0,
                    x: 540,
                    width: 100,
                    height: 420,
                    items : [],
                    border: false,
                }, {   xtype: 'multislider',
                    id: 'slider_' + python_varname,
                    minValue: 0,
                    maxValue: 100,
                    increment: 0.1,
                    x: 100, y: 410,
                    width: 400,
                    listeners: {
                        /* Only changecomplete; don't want too many render
                        events */
                        changecomplete: function(slider, newValue, thumb) {
                            yt_rpc.ExtDirectREPL.execute(
                                {code:python_varname + ".scroll_zoom(" +
                                      newValue + ")",
                                 hide:true}, cell_finished);
                        }
                    }
                },{
                    xtype: 'combo',
                    text: 'Field',
                    x: 100,
                    y: 435,
                    width: 400,
                    store:widget_data['fields'],
                    value:widget_data['initial_field'],
                    editable: false,
                    triggerAction: 'all',
                    validateOnBlur: false,
                    listeners: {select: function(combo, record, index) {
                        var newValue = record.data['field1'];
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.set_current_field("' +
                                newValue + '")', hide:false},
                            cell_finished);
                    }}
                }, {
                /* the single buttons for 10% pan*/
                    xtype:'button',
                    iconCls: 'singleuparrow',
                    id: 'singleuparrow',
                    //text: 'North',
                    x: 40,
                    y: 10,
                    handler: function(b,e) {
                        cc = python_varname + '.pan_rel((0.0, -0.1))'
                        yt_rpc.ExtDirectREPL.execute(
                        {code:cc, hide:true}, cell_finished); 
                    }
                }, {
                    xtype:'button',
                    iconCls: 'singlerightarrow',
                    id: 'singlerightarrow',
                    //text:'East',
                    x : 60,
                    y : 30,
                    handler: function(b,e) {
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.pan_rel((0.1, 0.0))',
                             hide:true},
                        cell_finished); 
                    }
                }, {
                    xtype:'button',
                    iconCls: 'singledownarrow',
                    id: 'singledownarrow',
                    //text: 'South',
                    x: 40,
                    y: 50,
                    handler: function(b,e) {
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.pan_rel((0.0, 0.1))',
                             hide:true},
                        cell_finished); 
                    }
                }, {
                    xtype: 'button',
                    iconCls: 'singleleftarrow',
                    id: 'singleleftarrow',
                    //text: 'West',
                    x: 20,
                    y: 30,
                    handler: function(b,e) {
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.pan_rel((-0.1, 0.0))',
                             hide:true},
                        cell_finished); 
                    }
                }, 
                /* the double buttons for 50% pan*/
                {
                    xtype:'button',
                    iconCls: 'doubleuparrow',
                    id:'doubleuparrow',
                    //text: 'North',
                    x: 40,
                    y: 80,
                    handler: function(b,e) {
                        cc = python_varname + '.pan_rel((0.0, -0.5))'
                        yt_rpc.ExtDirectREPL.execute(
                        {code:cc, hide:true}, cell_finished); 
                    }
                }, {
                    xtype:'button',
                    iconCls: 'doublerightarrow',
                    id:'doublerightarrow',
                    //text:'East',
                    x : 60,
                    y : 100,
                    handler: function(b,e) {
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.pan_rel((0.5, 0.0))',
                             hide:true},
                        cell_finished); 
                    }
                }, {
                    xtype:'button',
                    iconCls: 'doubledownarrow',
                    //text: 'South',
                    id: 'doubledownarrow',
                    x: 40,
                    y: 120,
                    handler: function(b,e) {
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.pan_rel((0.0, 0.5))',
                             hide:true},
                        cell_finished); 
                    }
                }, {
                    xtype: 'button',
                    iconCls: 'doubleleftarrow',
                    id: 'doubleleftarrow',
                    //text: 'West',
                    x: 20,
                    y: 100,
                    handler: function(b,e) {
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.pan_rel((-0.5, 0.0))',
                             hide:true},
                        cell_finished); 
                    }
                },
                /* Now the zoom buttons */
                {
                    xtype: 'button',
                    text: 'Zoom In 10x',
                    id: "zoom10x",
                    x: 10,
                    y: 160,
                    width: 80,
                    handler: function(b,e) {
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.zoom(10.0)',
                             hide:true},
                        cell_finished); 
                    }
                },{
                    xtype: 'button',
                    text: 'Zoom In 2x',
                    id: "zoom2x",
                    x: 10,
                    y: 185,
                    width: 80,
                    handler: function(b,e) {
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.zoom(2.0)',
                             hide:true},
                        cell_finished); 
                    }
                },{
                    xtype: 'button',
                    text: 'Zoom Out 2x',
                    id:'zoomout2x',
                    x: 10,
                    y: 210,
                    width: 80,
                    handler: function(b,e) {
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.zoom(0.5)',
                             hide:true},
                        cell_finished); 
                    }
                },{
                    xtype: 'button',
                    text: 'Zoom Out 10x',
                    id:'zoomout10x',
                    x: 10,
                    y: 235,
                    width: 80,
                    handler: function(b,e) {
                        yt_rpc.ExtDirectREPL.execute(
                            {code:python_varname + '.zoom(0.1)',
                             hide:true},
                        cell_finished); 
                    }
                },{
                    xtype: 'button',
                    text: 'Upload Image',
                    x: 10,
                    y: 285,
                    width: 80,
                    tooltip: "Upload the current image to " +
                             "<a href='http://imgur.com'>imgur.com</a>",
                    handler: function(b,e) {
                        img_data = image_dom.src;
                        yt_rpc.ExtDirectREPL.upload_image(
                            {image_data:img_data,
                             caption:metadata_string},
                        function(rv) {
                            var alert_text;
                            if(rv['uploaded'] == false) {
                                alert_text = "Failure uploading image!";
                            } else {
                                alert_text = "Uploaded to " +
                                        rv['upload']['links']['imgur_page'];
                            }
                            Ext.Msg.alert('imgur.com', alert_text);
                            var record = new logging_store.recordType(
                                {record: alert_text });
                            logging_store.add(record, number_log_records++);
                        }); 
                    }
                },{
                    xtype: 'button',
                    text: 'Pannable Map',
                    x: 10,
                    y: 335,
                    width: 80,
                    tooltip: "Open a pannable map in a new tab",
                    handler: function(b,e) {
                        img_data = image_dom.src;
                        yt_rpc.ExtDirectREPL.create_mapview(
                            {widget_name:python_varname},
                        function(rv) {
                            /*alert(rv);*/
                        }); 
                    }
                },{
                    xtype: 'panel',
                    layout: 'vbox',
                    id: 'rhs_panel_' + python_varname,
                    width: 250,
                    height: 460,
                    x: 690, y: 10,
                    layoutConfig: {
                        align: 'stretch',
                        pack: 'start',
                    },
                    items: [
                        {
                          xtype: 'panel',
                          title: 'Plot MetaData',
                          id: 'metadata_' + python_varname,
                          style: {fontFamily: '"Inconsolata", monospace'},
                          html: 'Welcome to the Plot Window.',
                          height: 200,
                        }, {
                          xtype: 'panel',
                          title: 'Plot Editor',
                          id: 'plot_edit',
                          flex: 1,
                          items : [
                             {
                               text: 'Display',
                               x: 20,
                               y: 20,
                               width : 100,
                               xtype: 'combo',
                               editable: false,
                               triggerAction: 'all',
                               validateOnBlur: false,
                               store: ['log10', 'linear'],
                               value: widget_data['initial_transform'],
                               listeners: {select: function(combo, record, index){ 
                                   var newValue = '"' + record.data['field1'] + '"';
                                   yt_rpc.ExtDirectREPL.execute(
                                       {code:python_varname + '.set_transform('
                                         + python_varname + '._current_field, '
                                         + newValue + ')', hide:false},
                                         cell_finished);
                               }}
                             }
                          ]
                        }]
                }
            ]
        }
    );

    viewport.get("center-panel").activate("pw_" + this.id);
    viewport.get("center-panel").doLayout();
    viewport.doLayout();
    this.panel = viewport.get("center-panel").get("pw_" + python_varname);
    this.panel.doLayout();
    this.panel.show();
    this.image_panel = this.panel.get("image_panel_"+python_varname);
    this.ticks = this.panel.get("ticks_"+python_varname);
    var ticks = this.ticks;
    this.metadata_panel = this.panel.get("rhs_panel_" + python_varname).get("metadata_" + python_varname);
    this.zoom_scroll = this.panel.get("slider_" + python_varname);
    var image_dom = this.image_panel.el.dom;
    var control_panel = this.panel;
    var metadata_string;

    this.accept_results = function(payload) {
        this.image_panel.el.dom.src = "data:image/png;base64," + payload['image_data'];
        this.zoom_scroll.setValue(0, payload['zoom'], true);
        examine = this.metadata_panel;
        this.metadata_panel.update(payload['metadata_string']);
        metadata_string = payload['metadata_string'];
        ticks.removeAll();
        Ext.each(payload['ticks'], function(tick, index) {
            console.log(tick);
            ticks.add({xtype:'panel',
                       width: 10, height:1,
                       style: 'background-color: #000000;',
                       html:'&nbsp;',
                       x:0, y: 10 + tick[0]});
            ticks.add({xtype:'panel',
                       width: 90, height:15,
                       border: false,
                       style: 'font-family: "Inconsolata", monospace;' +
                              'font-size: 12px;',
                       html: '' + tick[2] + '',
                       x:12, y: 4 + tick[0]});
            examine = tick;
        });
        examine = payload['ticks'];
        ticks.doLayout();
    }

    yt_rpc.ExtDirectREPL.execute(
        {code:python_varname + '.zoom(1.0)', hide:true},
        cell_finished);
}

widget_types['plot_window'] = WidgetPlotWindow;
