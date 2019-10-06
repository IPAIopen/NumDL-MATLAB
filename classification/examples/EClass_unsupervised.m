close all; clear all;

r = [0 2 4];

fig1 = figure(1); clf;
fig1.Name = 'data';
hold on;

fig3 = figure(3); clf;
fig3.Name = 'semi';
hold on;

fig2 = figure(2); clf;
fig2.Name = 'labeled'
hold on;
for k=1:numel(r)
    alpha = rand(1,300);
    rad   = r(k)+rand(1,300);
    X = rad.*[cos(2*pi*alpha);sin(2*pi*alpha)];
    figure(1);
    plot(X(1,:),X(2,:),'.k','MarkerSize',10)


    figure(2)
    plot(X(1,:),X(2,:),'.','MarkerSize',10)
    
    figure(3);
    p=plot(X(1,:),X(2,:),'.k','MarkerSize',10)
    plot(X(1,1:3),X(2,1:3),'.','MarkerSize',50)

end
    
figDir = '/Users/lruthot/Dropbox/Projects/NumDL-CourseNotes/images/'

for k=1:3
    fig = figure(k)
    axis equal tight off
    set(gca,'FontSize',20)
    printFigure(gcf,fullfile(figDir,['unsupervised_' fig.Name '.png']))
end
