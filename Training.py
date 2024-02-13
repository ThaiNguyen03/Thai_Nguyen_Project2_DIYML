class Training:
    def _init_(self,user_id,project_id):
        self.user_id = user_id
        self.project_id = project_id
    def configure_training(self,parameters):
        self.parameters =parameters
    def start_training(self, model, data, parameters):
        model.fit(data);
    def get_training_stats(self):
        pass
    def run_iteration(self):
        pass