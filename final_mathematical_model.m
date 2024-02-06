try
    % Load data, preserving original column names
    opts = detectImportOptions('MIT_BIH_Arrhythmia_Database.csv', 'VariableNamingRule', 'preserve');
    data = readtable('MIT_BIH_Arrhythmia_Database.csv', opts);

    % Extract relevant features from the loaded data for a specific row (e.g., the first row)
    RR_input = data.('0_pre-RR')(32800);
    QRS_duration_input = data.('0_qrs_interval')(32800);
    QRS_amplitude_input = data.('0_rPeak')(327800) - data.('0_qPeak')(32800);

    % Calculate mean and standard deviation for each feature
    mean_RR = mean(data.('0_pre-RR'));
    std_RR = std(data.('0_pre-RR'));

    mean_QRS_duration = mean(data.('0_qrs_interval'));
    std_QRS_duration = std(data.('0_qrs_interval'));

    mean_QRS_amplitude = mean(data.('0_rPeak') - data.('0_qPeak'));
    std_QRS_amplitude = std(data.('0_rPeak') - data.('0_qPeak'));

    % Set dynamic thresholds (e.g., 1 standard deviation away from the mean)
    RR_threshold = mean_RR + std_RR;
    QRS_duration_threshold = mean_QRS_duration + std_QRS_duration;
    QRS_amplitude_threshold = mean_QRS_amplitude - std_QRS_amplitude;

    % Check user-input data against thresholds
    is_arrhythmia = (RR_input > RR_threshold) | (QRS_duration_input > QRS_duration_threshold) | (QRS_amplitude_input < QRS_amplitude_threshold);

    % Display the result
    if is_arrhythmia
        disp('Arrhythmia detected!');
    else
        disp('Normal heartbeat detected.');
    end
catch ME
    % Display detailed error message
    disp(getReport(ME));
end
