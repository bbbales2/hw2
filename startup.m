fprintf('This is /Users/gilbert/Documents/CS240aWinter2014/Assignments/hw2/matlab/startup.m\n');

dfile = sprintf('diary-%s.txt',date);
fprintf('Starting diary file %s\n',dfile);
diary(dfile);