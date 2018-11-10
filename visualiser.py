from collections import defaultdict
import visdom

# Do not print error messages if visdom client throws error
IGNORE_VISDOM_ERROR = 10

class VisdomWriter:
    """Class to publish data to Visdom"""
    def __init__(self, enabled=True, logger=None):
        """
        Initialise writer

        Parameters
        ----------
        enabled (bool): Whether or not to publish data to visdom
        """
        self.vis = visdom.Visdom()
        self.enabled = enabled
        self.logger = logger
        self.xdict = defaultdict(list)
        self.ydict = defaultdict(list)
        self.visdom_error_count = 0

    def text(self, data, title="default"):
        """
        Publish text data to visdom

        Parameters
        ----------
        data (string): text data to publish
        title (string): visdom window title
        """
        if self.enabled:
            try:
                self.vis.text(data, win=title, opts=dict(title=title))  
                self.visdom_error_count = 0
            except Exception as e:
                self.visdom_error_count += 1
                if (self.visdom_error_count < IGNORE_VISDOM_ERROR):
                    if self.logger:
                        self.logger.error('Failed to publish text to visdom server: {}'.format(e))
                    else:
                        print('Failed to publish text to visdom server: {}'.format(e))                            

        if self.logger:
            self.logger.debug('{}: {}'.format(title, data))

    def push(self, item, key="default"):
        """
        Publish data required to plot line chart, useful for plotting losses

        Parameters
        ----------
        item (object): number or tensor
        key (string): unique name of item being plotted (e.g "actor")
        """
        if self.enabled:
            self.ydict[key].append(item)
            self.xdict[key].append(len(self.ydict[key]))            
            try:
                self.vis.line(Y=self.ydict[key], X=self.xdict[key], win=key, opts=dict(title=key))
                self.visdom_error_count = 0
            except Exception as e:
                self.visdom_error_count += 1
                if self.visdom_error_count < IGNORE_VISDOM_ERROR:
                    if self.logger:
                        self.logger.error('Failed to publish data to visdom server: {}'.format(e))
                    else:
                        print('Failed to publish data to visdom server: {}'.format(e))

        if self.logger:
            self.logger.debug('{}: {}'.format(key, item))





        
