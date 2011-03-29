var WidgetPlotWindow = function(python_varname) {
    this.id = python_varname;
    this.print_python = function(b, e) {
        yt_rpc.ExtDirectREPL.execute({code:'print "' + python_varname + '"'},
                                     function(f, a)
                                     {alert(a.result['output']);});
    }

    viewport.get("center-panel").add(
                  {
                  xtype: 'panel',
                  id: "pw_" + this.id,
                  title: this.id,
                  iconCls: 'graph',
                  autoScroll: true,
                  layout:'absolute',
                  items: [ 
                      {xtype:'panel',
                       id: 'image_panel_' + this.id,
                       autoEl: {
                         tag: 'img',
                         id: "img_" + this.id,
                         width: 400,
                         height: 400,
                       },
                       x: 10,
                       y: 10,
                       width: 400,
                       height: 400,
                      },
                      {xtype:'button',
                       text: 'North',
                       x: 205,
                       y: 10},
                      {xtype:'button',
                       text:'East',
                       x : 410,
                       y : 205,
                       handler: this.print_python},
                      {xtype:'button',
                       text: 'South',
                       x: 205,
                       y: 410},
                      {xtype: 'button',
                       text: 'West',
                       x: 10,
                       y: 205},
                       ]
                  });

    viewport.doLayout();
    this.panel = viewport.get("center-panel").get("pw_" + python_varname);
    this.panel.doLayout();
    this.image_panel = this.panel.get("image_panel_"+python_varname);

    this.handle_payload = function(payload) {
        this.image_panel.el.dom.src = "data:image/png;base64," + payload['image_data'];
    }
}

widget_types['plot_window'] = WidgetPlotWindow;
