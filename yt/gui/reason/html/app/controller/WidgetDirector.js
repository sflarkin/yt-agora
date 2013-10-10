/**********************************************************************
Widget controller class

Copyright (c) 2013, yt Development Team.

Distributed under the terms of the Modified BSD License.

The full license is in the file COPYING.txt, distributed with this software.
***********************************************************************/

Ext.define('Reason.controller.WidgetDirector', {
    extend: 'Ext.app.Controller',
    requires: ["Reason.controller.widgets.SampleWidget",
               "Reason.controller.widgets.PlotWindow",
               "Reason.controller.widgets.ProgressBar",
               "Reason.controller.widgets.GridDataViewer",
               "Reason.controller.widgets.ParameterFile",
               "Reason.controller.widgets.PannableMap",
               "Reason.controller.widgets.PhasePlot",
               "Reason.controller.widgets.Scene",
    ],
    stores: ['WidgetTypes', 'WidgetInstances'],
    views: ['WidgetTypesGrid', 'WidgetInstancesGrid'],

    init: function() {
        Ext.iterate(Reason.controller.widgets, function(i, w, ws) {
            Ext.require(w.getName());
            this.registerWidget(w);
        }, this);
        this.application.addListener({
            createwidget: {fn: this.createWidget, scope: this},
            showwidgets: {fn: this.showWidgetMenu, scope: this},
            payloadwidget: {fn: this.newWidgetCreated, scope: this},
            payloadwidget_payload: {fn: this.sendPayload, scope: this},
            enabledebug: {fn: this.enableDebug, scope: this},
        });
        this.callParent(arguments);
    },

    registerWidget: function(w) {
        if (w.widgetName == null) {return;}
        console.log("Registering " + w.widgetName);
        this.getWidgetTypesStore().add({
                   widgetname: w.widgetName,
                   widgetclass: w,
                   displayname: w.displayName,
                   pfs: w.supportsParameterFiles,
                   objs: w.supportsDataObjects,
        });
    },

    createWidget: function(b, e) {
        var w = b.widget;
        console.log("Asked to create " + b.widget.widgetName);
        b.widget.preCreation(b.dataObj);
    },

    showWidgetMenu: function(treerecord, e) {
        var contextMenu = Ext.create('Ext.menu.Menu', {plain: true,});
        var data = treerecord.data;
        var w;
        this.getWidgetTypesStore().each(function(record, idx) {
            w = record.data;
            if (((data.type == 'parameter_file') && (w.pfs  == false)) 
             || ((data.type != 'parameter_file') && (w.objs == false))) {
              return;
            }
            contextMenu.add({xtype:'menuitem',
                             text: w.displayname,
                             listeners: {
                                click: {
                                    fn : this.createWidget,
                                    scope: this
                                },
                             },
                             widget: w.widgetclass,
                             dataObj: data
            });
        }, this);
        contextMenu.showAt(e.getXY());
    },

    newWidgetCreated: function(payload) {
        /* We have the following fields:
                type             ('widget')
                widget_type
                varname
                data             (has subfields)

           We now obtain our class, create that with the factory, and we add
           the resultant class to our store.
        */
        var resultId = this.getWidgetTypesStore().find(
            'widgetname', payload['widget_type']);
        if (resultId == -1) {
            Ext.Error.raise('Did not recognize widget type "' +
                            payload['widget_type'] + '".');
        }
        var widgetInfo = this.getWidgetTypesStore().getAt(resultId).data;
        /* The widget adds its view to the viewport. */
        var newWidget = Ext.create(widgetInfo['widgetclass'].getName(),
                            {payload: payload});
        console.log("Adding widget payload with varname " + payload['varname']);
        this.getWidgetInstancesStore().add({
            widgetid: payload['varname'],
            widgettype: widgetInfo.widgetname,
            widget: newWidget
        });
        Ext.ComponentQuery.query("viewport > #center-panel")[0].add(
            newWidget.createView());
    },

    sendPayload: function(payload) {
        var resultId = this.getWidgetInstancesStore().find(
            'widgetid', payload['widget_id']);
        if (resultId == -1) {
            Ext.Error.raise('Could not find widget "' +
                            payload['widget_id'] + '".');
        }
        /*console.log("Directing payload for " + payload['widget_id'] +
                    " to resultId " + resultId);*/
        if (payload['binary'] != null) {
            this.loadBinaryData(payload);
            return;
        }
        var widgetInfo = this.getWidgetInstancesStore().getAt(resultId).data;
        widgetInfo['widget'].applyPayload(payload);
    },

    loadBinaryData: function(payload1) {
        /* https://developer.mozilla.org/en/using_xmlhttprequest
           including progress */
        function loadBinaryPayload(payload) {
            var req = new XMLHttpRequest();
            var bkeys = payload['binary'];
            var nLeft = bkeys.length;
            var bkey = bkeys[nLeft - 1];
            var director = this;
            payload['binary'] = null;
            req.open("GET", bkey[1], true);
            req.responseType = "arraybuffer";
            onLoad = function(e) {
                payload[bkey[0]] = req.response;
                nLeft = nLeft - 1;
                if (nLeft == 0) {
                  director.sendPayload(payload);
                } else {
                  bkey = bkeys[nLeft - 1];
                  req.open("GET", bkey[1], true);
                  req.responseType = "arraybuffer";
                  req.onload = onLoad;
                  req.send();
                  exaine = payload;
                }
            }
            req.onload = onLoad;
            req.send();
        }
        loadBinaryPayload.call(this, payload1);
    },

    enableDebug: function() {
        if(this.instanceView) {return;}
        this.instanceView = Ext.widget('widgetinstancesgrid');
        this.typeView = Ext.widget('widgettypesgrid');
        Ext.ComponentQuery.query("viewport > #center-panel")[0].add(
            this.instanceView);
        Ext.ComponentQuery.query("viewport > #center-panel")[0].add(
            this.typeView);
    },

});
