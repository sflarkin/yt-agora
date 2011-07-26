/**********************************************************************
The main GUI facility for Reason

Author: Cameron Hummels <chummels@gmail.com>
Affiliation: Columbia
Author: Jeffrey S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Author: Britton Smith <brittonsmith@gmail.com>
Affiliation: MSU
Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Homepage: http://yt.enzotools.org/
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


var viewport;
var widget_types = {}
var widget_list = {}

var examine;
var number_log_records = 0;
var number_images = 0;

var res;
var cell_count = 0;

var handle_result = function(f, a) {
    if(a.status == false){
        Ext.Msg.alert("Error", "Something has gone wrong.");
        examine = {f: f, a: a};
        return;
    }
    cell_finished(a.result);
}

var handle_payload = function(pp) {
    cell_finished(pp);
}

var repl_input = new Ext.form.Panel({
    title: 'YT Input',
    url: 'push',
    flex: 1.0,
    layout: 'fit',
    padding: 5,
    height: '100%',
    flex: 1.0,
    items: [{
        id: 'input_line',
        xtype: 'textarea',
        width: '100%',
        autoScroll: true,
        name: 'line',
        allowBlank: 'True',
        bodyStyle: 'font-family: "monospace";',
        listeners: {
            specialkey: function(f, e){
                if (e.getKey() == e.ENTER) {
                    disable_input();
                    yt_rpc.ExtDirectREPL.execute({
                        code:repl_input.getComponent('input_line').getValue()},
                    handle_result);
                }
            },
            afterrender: function(f, e){
                //var input_line_drop_target_el = repl_input.get("input_line").el.dom;
                var input_line_drop_target_el = repl_input.body.dom;

                var input_line_drop_target = new Ext.dd.DropTarget(input_line_drop_target_el, {
                    ddGroup     : 'pfDDgroup',
                    notifyEnter : function(ddSource, e, data) {
                        repl_input.body.stopFx();
                        repl_input.body.highlight();
                    },
                    notifyDrop  : function(ddSource, e, data){

                        var varname = data.node.attributes.objdata.varname;
                        /* There is possibly a better way to do this, where it's also inserted correctly. */
                        var line = repl_input.getComponent("input_line");
                        line.setValue(line.getValue() + varname);
                        line.focus();
                        return(true);
                    }
                });
            },
        },
    },],
});


var CellInputContainer = new Ext.Panel({
    title: 'YT Input',
    flex: 0.3,
    layout: {type: 'hbox',
             pack: 'start',
             align: 'stretch',
             },
    items: [ repl_input,
            { xtype: 'button',
              width: 24,
              height: 24,
              iconCls: 'doubledownarrow',
              tooltip: 'Execute Cell',
              listeners: {
                  click: function(f, e) {
                    disable_input();
                    yt_rpc.ExtDirectREPL.execute({
                        code:repl_input.get('input_line').getValue()},
                    handle_result);
                  }
              },
            }
           ]
});


var OutputContainer = new Ext.Panel({
    title: 'YT Output',
    id: 'output_container',
    autoScroll: true,
    flex: 4.0,
    items: []
});

var examine;
var notebook;

/*var treePanel = new Ext.tree.TreePanel({
    iconCls: 'nav',
    id: 'tree-panel',
    layout: 'anchor',
    region:'west',
    split: true,
    anchor: '100% -35',
    minSize: 150,
    autoScroll: true,
    rootVisible: false,
    ddGroup: 'pfDDgroup',
    enableDD: true,
    root:new Ext.tree.TreeNode({
        expanded:true,
        leaf:false,
        text:''
    }),
    listeners: {
        render: {
            fn: function() {
                Ext.getBody().on("contextmenu", Ext.emptyFn,
                null, {preventDefault: true});
            }
        },
        dblclick: {
            fn: function(node, e) {
                treePanel.fireEvent("contextmenu", node, e);
            }
        },
        contextmenu: {
            fn: function(node, event){
                var rightclickMenu;
                if (node.attributes.objdata.type == 'obj') {
                  rightClickMenu = new Ext.menu.Menu({
                      items: [
                          {
                              text: 'Phase Plot',
                              handler: getPhasePlotHandler(node),
                          }, 
                      ]
                  });
                } else if (node.attributes.objdata.type == 'pf') {
                  rightClickMenu = new Ext.menu.Menu({
                      items: [
                          /*{
                              text: 'View Grids',
                              handler: getGridViewerHandler(node),
                          },*/ {
                              text: 'View Grid Data',
                              handler: getGridDataViewerHandler(node),
                          }, {
                              text: 'Open slice',
                              handler: getSliceHandler(node),
                          }, {
                              text: 'Open projection',
                              handler: getProjectionHandler(node),
                          /*}, {
                              text: 'Create Sphere',
                              handler: getSphereCreator(node), */
                          }, /*{
                              text: 'View Streamlines',
                              handler: getStreamlineViewerHandler(node),
                          }, */
                      ]
                  });
                }
                rightClickMenu.showAt(event.xy);
            }
        }
    }
});*/

var status_panel;

Ext.define("log", {extend: "Ext.data.Model", fields: ['record']});

var logging_store = new Ext.data.Store({
    model: 'log',
    data: [],
    proxy: {
        type: 'memory',
        reader: {
            type: 'json',
            root: 'users'
        }
    }
});

var heartbeat_request = false;
var task_runner = new Ext.util.TaskRunner();
var heartbeat;


Ext.onReady(function(){
    Ext.BLANK_IMAGE_URL = 'resources/resources/images/default/s.gif';

    // NOTE: This is an example showing simple state management. During development,
    // it is generally best to disable state management as dynamically-generated ids
    // can change across page loads, leading to unpredictable results.  The developer
    // should ensure that stable state ids are set for stateful components in real apps.
    // it's a cold day for pontooning.
    Ext.state.Manager.setProvider(new Ext.state.CookieProvider());

    // Go ahead and create the TreePanel now so that we can use it below
    viewport = new Ext.container.Viewport({
        layout: 'border',
        items: [
		// lazily created panel (xtype:'panel' is default)
            {
                xtype: 'grid',
                store: logging_store,
                defaults: { width: 800 },
                columns: [ {
                     header: 'Log Message',
                     dataIndex: 'record',
                     flex:1.0,
                     sortable: false,
                     width:800} ],
                autofill: true,
                region: 'south',
                id: "status-region",
                cls: "status-logger",
                split: true,
                height: 100,
                maxSize: 200,
                collapsible: true,
                title: 'Status',
                margins: '0 0 0 0',
            }, {
                region: 'west',
                id: 'west-panel', // see Ext.getCmp() below
                title: 'Data Objects',
                split: true,
                width: 200,
                minSize: 175,
                maxSize: 400,
                collapsible: true,
                margins: '0 0 0 5',
                layout: {
                    type: 'anchor',
                },
                items: [{
                        xtype: 'toolbar',
                        items: [ main_menu ],
                    },
                ]
		  // in this instance the TabPanel is not wrapped by another panel
		  // since no title is needed, this Panel is added directly
		  // as a Container
            },{
                xtype: 'tabpanel',
                region: 'center', 
                id: 'center-panel',
                deferredRender: false,
                activeTab: 0,     
                items: [{
                        title: 'YT',
                        id: 'notebook',
                        layout: {
                                  type:'vbox',
                                  align:'stretch',
                                  pack:'start'
                                },
                        closable: false,
                        autoScroll: false,
                        iconCls: 'console',
                        items: [CellInputContainer, OutputContainer]
                    }, 
                ]
            }
        ]
    });

// get a reference to the HTML element with id "hideit" and add a click listener to it 
    if (typeof(console) != "undefined") {
        console.log('Mitchell!\nPardon me! Mitchell!')
    }
    Ext.get("hideit").on('click', function(){
// get a reference to the Panel that was created with id = 'west-panel' 
	    var w = Ext.getCmp('west-panel');
// expand or collapse that Panel based on its collapsed property state
// need to make room for six sour cream burritos
        w.collapsed ? w.expand() : w.collapse();
    });
    
    notebook = viewport.getComponent("center-panel").getComponent("notebook");
    status_panel = viewport.getComponent("status-region").getComponent("status-div");
    logging_store.add({record: 'Welcome to yt.'});
    logging_store.add({record: 'After entering a line of code in the YT' +
                      'Input field, press shift-enter to evaluate.' });
    logging_store.add({record: '4d3d3d3 engaged.' });

    if (!Ext.state.Manager.get("reason_welcomed", false)) {
        Ext.MessageBox.alert("Reason v0.5",
        "Welcome to Reason.  <br>Treat the 'YT Input' field as a YT/python intepreter.<br>Press shift-enter to evaluate.",
        function(b,e){ repl_input.getComponent("input_line").focus(); });
        Ext.state.Manager.set("reason_welcomed", true);
    } else { 
        repl_input.getComponent("input_line").focus();
    }

    /* Set up the heartbeat */
    var num = 0;
    heartbeat = {
    run:
      function(){ if (heartbeat_request == true) return; 
        heartbeat_request = true;
        yt_rpc.ExtDirectREPL.heartbeat(
            {}, function(f, a) {
            heartbeat_request = false;
            if (f != null) {
                handle_result(f, a);
            }})},
    interval: 250};

    task_runner.start(heartbeat);
                         
});
