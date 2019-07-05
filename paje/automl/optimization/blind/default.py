from paje.automl.optimization.blind.random import RandomAutoML


class DefaultAutoML(RandomAutoML):
    """
    Always select default hyperparameters.
    """

    def next_pipelines(self, data):
        """ TODO the docstring documentation
        """
        components = self.choose_modules()
        self.curr_pipe = Pipeline(components, show_warns=self.show_warns,
                                  storage=self.storage_for_components)
        tree = self.curr_pipe.tree()

        try:
            args = tree.tree_to_config()
        except SamplingException as exc:
            print(' ========== Pipe:\n', self.curr_pipe)
            raise Exception(exc)

        args['random_state'] = self.random_state
        self.curr_pipe = self.curr_pipe.build(**args)
        return [self.curr_pipe]