import os
from flask import Flask, render_template, request, url_for, redirect
from flask_sqlalchemy import SQLAlchemy

from sqlalchemy.sql import func
class Login:
    def login(self,username,password):
        # check user login with database(later)
        pass
    def generate_api_key(self):
        pass