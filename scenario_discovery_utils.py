import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from textwrap import wrap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from itertools import product
from seaborn import heatmap
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn import tree
import graphviz
import os
import plotly
import plotly.express as px
import string
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import host_subplot
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from colour import Color

class GlobalProperties:
    def __init__(self):
        self.supported_input_scenarios = ['GLB_GRP_NORM', 'GLB_RAW', 'USA', 'CHN', 'EUR', 'CHN_ENDOG_RENEW', 'CHN_ENDOG_EMISSIONS']
        self.supported_output_scenarios = ['REF_GLB_RENEW_SHARE', 'REF_USA_RENEW_SHARE', 'REF_CHN_RENEW_SHARE', 'REF_EUR_RENEW_SHARE',
            '2C_GLB_RENEW_SHARE', '2C_USA_RENEW_SHARE', '2C_CHN_RENEW_SHARE', '2C_EUR_RENEW_SHARE', 'REF_GLB_RENEW', 'REF_GLB_TOT', '2C_GLB_RENEW', '2C_GLB_TOT',
            '2C_CHN_ENDOG_RENEW', 'REF_CHN_EMISSIONS']
        self.colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']

class Preparation(GlobalProperties):
    def __init__(self, input_case, output_case):
        super().__init__()
        """Creates an object that stores the input and output dataframes that contain simulation results. These objects have a number of
        methods that are helpful for scenario discovery analysis. Be careful not to mix incompatible input/output cases together (e.g.
        using U.S. input data to analyze share of renewables in China).

        Args:
            input_case (str): the input data to be used (supported: GLB_GRP_NORM (global, grouped, and normed); GLB_RAW
            (global, full number of variables, no normalization); USA (full number of variables with GDP + Pop specific to US);
            CHN (full number of variables with GDP + Pop specific to China); EUR (full number of variables with GDP + Pop
            specific to EU); CHN_ENDOG_RENEW (output-output mapping as shown in paper section 4.1); CHN_ENDOG_EMISSIONS (output-output
            mapping as shown in paper section 4.2))

            output_case (str): the output metric (supported: REF_GLB_RENEW_SHARE (global share of renewables under
            the reference scenario); REF_USA_RENEW_SHARE (US share of renewables under the reference scenario); REF_CHN_RENEW_SHARE
            (China share of renewables under the reference scenario); REF_EUR_RENEW_SHARE (EU share of renewables under the reference
            scenario); 2C_GLB_RENEW_SHARE (global share of renewables under the policy scenario); 2C_USA_RENEW_SHARE (US share of
            renewables under the policy scenario); 2C_CHN_RENEW_SHARE (China share of renewables under the policy scenario);
            2C_EUR_RENEW_SHARE (EU share of renewables under the policy scenario); REF_GLB_RENEW (global renewable energy
            production in Twh); REF_GLB_TOT (total global energy production in Twh); 2C_GLB_RENEW (global renewable energy production
            in Twh under policy); 2C_GLB_TOT (total global energy production in Twh under policy); 2C_CHN_ENDOG_RENEW (output-output 
            mapping as shown in paper section 4.1); REF_CHN_EMISSIONS (output-output mapping as shown in paper section 4.1); CHN_ENDOG_EMISSIONS 
            (output-output mapping as shown in paper section 4.2))

        Raises:
            ValueError: if an invalid input case is passed
            ValueError: if an invalid output case is passed
        """
        self.input_case = input_case
        self.output_case = output_case
        self.hyperparams = None
        # "translates" between the abbreviations used in the code and the long forms used in the paper
        self.readability_dict = {'GLB_RAW': 'Global', 'REF': 'Share of Renewables Under Reference', 'POL': 'Share of Renewables Under Policy',
                                'CHN': 'China', 'USA': 'USA', 'EUR': 'Europe'}
        self.ref_or_pol = 'REF' if 'REF' in self.output_case else 'POL'

        self.natural_to_code_conversions_dict_inputs = {'GLB_GRP_NORM': ['samples-norm+groupedav', 'A:J'], 'GLB_RAW': ['samples', 'A:BB'], 'USA': ['samples', 'A:AZ, BC:BF'],
            'CHN': ['samples', 'A:AZ, BG:BJ'], 'EUR': ['samples', 'A:AZ, BK: BN'], 'CHN_ENDOG_RENEW': ['2C_CHN_renew_outputs_inputs', 'A:G'],
            'CHN_ENDOG_EMISSIONS': ['REF_CHN_emissions_inputs', 'A:H']}
        if self.input_case in self.supported_input_scenarios:
            sheetname = self.natural_to_code_conversions_dict_inputs[self.input_case][0]
            columns = self.natural_to_code_conversions_dict_inputs[self.input_case][1]
            if self.input_case == "GLB_GRP_NORM":
                self.input_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = columns, nrows = 400, engine = 'openpyxl')
            else:
                self.input_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = columns, nrows = 400, header = 2, engine = 'openpyxl')
            
            if self.input_case == 'CHN_ENDOG_RENEW':
                # these scenarios already have the crashed runs removed
                pass
            else:
                if '2C' in output_case:
                    # indicates a policy scenario in which runs crashed
                    crashed_run_numbers = [3, 14, 74, 116, 130, 337]
                    self.input_df = self.input_df.drop(index = [i - 1 for i in crashed_run_numbers])
        else:
            raise ValueError('This input scenario is not supported. Supported scenarios are {}'.format(self.supported_input_scenarios))

        self.natural_to_code_conversions_dict_outputs = {'REF_GLB_RENEW_SHARE': 'ref_GLB_renew_share', 'REF_USA_RENEW_SHARE': 'ref_USA_renew_share',
            'REF_CHN_RENEW_SHARE': 'ref_CHN_renew_share', 'REF_EUR_RENEW_SHARE': 'ref_EUR_renew_share', '2C_GLB_RENEW_SHARE': '2C_GLB_renew_share',
            '2C_USA_RENEW_SHARE': '2C_USA_renew_share', '2C_CHN_RENEW_SHARE': '2C_CHN_renew_share', '2C_EUR_RENEW_SHARE': '2C_EUR_renew_share',
            'REF_GLB_RENEW': 'ref_GLB_renew', 'REF_GLB_TOT': 'ref_GLB_total_elec', '2C_GLB_RENEW': '2C_GLB_renew', '2C_GLB_TOT': '2C_GLB_total_elec',
            '2C_CHN_ENDOG_RENEW': '2C_CHN_renew_outputs_output', 'REF_CHN_EMISSIONS': 'REF_CHN_emissions_output'}
        if self.output_case in self.supported_output_scenarios:
            sheetname = self.natural_to_code_conversions_dict_outputs[output_case]
            self.output_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = 'D:X', nrows = 400, engine = 'openpyxl')
            if self.output_case == '2C_CHN_ENDOG_RENEW' or self.output_case == 'REF_CHN_EMISSIONS':
                self.output_df = pd.read_excel('Full Data for Paper.xlsx', sheetname, usecols = 'A:B', nrows = 400, engine = 'openpyxl')
                print('Note: some methods are not supported for this output scenario because it contains data from only one year of the simulation (2050).')
            else:
                if '2C' in output_case:
                    # indicates a policy scenario in which runs crashed
                    crashed_run_numbers = [3, 14, 74, 116, 130, 337]
                    self.output_df = self.output_df.drop(index = [i - 1 for i in crashed_run_numbers])
        else:
            raise ValueError('This output scenario is not supported. Supported scenarios are {}'.format(self.supported_output_scenarios))

    def get_X(self, runs = False):
        """Get the exogenous dataset (does not include run numbers).

        Returns:
            DataFrame: Input variables and their values
        """
        if runs:
            return self.input_df
        else:
            return self.input_df[self.input_df.columns[1:]]

    def get_y(self, runs = False):
        """Get the endogenous dataset (does not include run numbers).

        Returns:
            DataFrame: Output timeseries
        """
        if runs:
            return self.output_df
        else:
            return self.output_df[self.output_df.columns[1:]]

    def get_y_by_year(self, year):
        """Get the series for an individual year.

        Args:
            year (int): A year included in the dataset (options:
            2007, and 2010-2100 in 5-year increments)

        Returns:
            Series: A pandas Series object with the data from the given year
        """
        return self.output_df[str(year)]

class Analysis:
    pass

class Visualization(GlobalProperties):
    def __init__(self, plot_type, sd_obj, display = True):
        super().__init__()
        self.plot_type = plot_type
        self.display = display
        self.sd_obj = sd_obj
        self.plot_functions_map = {"Distribution": self.distribution}

    def makePlot(self, *args):
        return self.plot_functions_map[self.plot_type](*args)

    def distribution(self, inputs_to_visualize = None):
        # visualize inputs as a histogram
        if not inputs_to_visualize:
            inputs_to_visualize = self.sd_obj.get_X().columns
        input_fig = make_subplots(rows = 1, cols = len(inputs_to_visualize))
        for i, input in enumerate(inputs_to_visualize):
            trace = go.Histogram(x = self.sd_obj.get_X()[input], name = input)
            input_fig.add_trace(trace, row = 1, col = i + 1)

        # visualize outputs as time series
        output_results = self.sd_obj.get_y()
        output_plot_df = pd.DataFrame(index = output_results.columns, columns = ["95th", "Median", "5th"])
        output_plot_df["95th"] = output_results.apply(lambda x: np.percentile(x, 95))
        output_plot_df["5th"] = output_results.apply(lambda x: np.percentile(x, 5))
        output_plot_df["Median"] = output_results.apply(lambda x: np.percentile(x, 50))

        output_fig = go.Figure([
            go.Scatter(
                name = 'Median',
                x = output_plot_df.index,
                y = output_plot_df['Median'],
                mode = 'lines',
                line = dict(color = self.colors[0]),
            ),
            go.Scatter(
                name = '95th Percentile',
                x = output_plot_df.index,
                y = output_plot_df['95th'],
                mode = 'lines',
                marker = dict(color = self.colors[1]),
                line = dict(width = 0),
                showlegend = False
            ),
            go.Scatter(
                name = '5th Percentile',
                x = output_plot_df.index,
                y = output_plot_df['5th'],
                marker = dict(color = self.colors[1]),
                line = dict(width = 0),
                mode = 'lines',
                # inelegant method to get a more transparent color for fill
                fillcolor = "rgba" + str((*Color(self.colors[1]).rgb, 0.2)),
                fill = 'tonexty',
                showlegend = False
            )
        ])
        output_fig.update_layout(
            yaxis_title = self.sd_obj.output_case,
            title = self.plot_type + " for " + self.sd_obj.output_case,
            hovermode = "x"
        )

        return input_fig, output_fig

    def parallelPlot(self, data_to_visualize, year_for_output, target):
        pass

class Display(GlobalProperties):
    def __init__(self, display = True):
        super().__init__()
        if display:
            with st.sidebar:
                st.header("Input and Ouput Data Info")
                st.subheader("Inputs")
                st.markdown("**GLB_RAW**: All of the indepedent (exogenous) variables, sampled independently from distributions specific to each variable\
                        and without any pre-processing steps (e.g. normalization) applied.")

    def displayCorrectOutputs(self, selected_input):
        region = selected_input.split('_')[0]
        output_options = [x for x in self.supported_output_scenarios if region in x]

        return output_options

    def displayDataMenu(self):
        st.write("Please choose the input and output data you'd like to analyze.")
        input_data = st.selectbox("Please select the input data (see sidebar for descriptions).",
                    (x for x in self.supported_input_scenarios))
        
        output_data = st.selectbox("Please select the output data (see sidebar for descriptions).",
                self.displayCorrectOutputs(input_data))

        return input_data, output_data

    def displayPlotsMenu(self):
        st.write("Please choose the type of plot you wish to generate.")
        plot = st.selectbox("Plots", ["Parallel Plot", "Top Features Plot", "Distribution"])

        return plot

    def makeSelectedPlot(self, plot_type, input_data, output_data):
        sd_obj = Preparation(input_data, output_data)
        if plot_type == "Distribution":
            with st.container():
                st.write("Select which inputs you'd like to visualize (max 5). Output data will be presented automatically.")
                inputs_to_visualize = st.multiselect("Inputs", sd_obj.get_X().columns, max_selections = 6)

                confirm = st.button("Confirm")
            
                if confirm:
                    input_fig, output_fig = Visualization(plot_type, sd_obj).makePlot(inputs_to_visualize)
                    st.plotly_chart(input_fig, use_container_width = True)
                    st.plotly_chart(output_fig, use_container_width = True)

    def run(self):
        with st.container():
            input_data, output_data = self.displayDataMenu()
            selected_plot = self.displayPlotsMenu()

            self.makeSelectedPlot(selected_plot, input_data, output_data)

# suggested workflow
# 1. pass in region, policy to work with
    # issue here: comparison across different scenario types
        # can there be an option to grid the layout and choose which plot can go in which grid?

if __name__ == "__main__":
    pd.options.plotting.backend = "plotly"
    st.set_page_config(layout = 'wide')

    elements = Display().run()
    # Visualization("Distribution", Preparation("GLB_RAW", "2C_GLB_RENEW_SHARE")).makePlot()