var gulp = require('gulp');
var babel = require('gulp-babel');
var sourcemaps = require('gulp-sourcemaps');
var uglify = require('gulp-uglify');

gulp.task('build', function() {
    return gulp.src('src/js/*.js')
        .pipe(babel({ presets: ['es2015'] }))
        .pipe(sourcemaps.init({ loadMaps: true }))
        .pipe(uglify())
        .pipe(sourcemaps.write())
        .pipe(gulp.dest('static/js'));
});

gulp.task('watch', function() {
    gulp.watch('src/js/*.js', ['build']);
});

gulp.task('default', ['build']);
